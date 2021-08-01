/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <algorithm>

// #include "comm.h"
// #include "core.h"
// #include "socket.h"
// #include "net.h"
#include "param.h"

#include <pthread.h>
#include <stdlib.h>
#include <poll.h>
#include <limits.h>
#include <fcntl.h>
#include <errno.h>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <net/if.h>

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SLEEP_INT            1000 // connection retry sleep interval in usec
#define RETRY_REFUSED_TIMES   2e4 // connection refused retry times before reporting a timeout (20 sec)
#define RETRY_TIMEDOUT_TIMES    3 // connection timed out retry times (each one can take 20s)
#define SOCKET_NAME_MAXLEN (NI_MAXHOST+NI_MAXSERV)

#define NCCL_SOCKET_SEND 0
#define NCCL_SOCKET_RECV 1

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2

#define NCCL_NET_HANDLE_MAXSIZE 64
#define MAX_REQUESTS NCCL_NET_MAX_REQUESTS

typedef enum {NCCL_LOG_NONE=0, NCCL_LOG_VERSION=1, NCCL_LOG_WARN=2, NCCL_LOG_INFO=3, NCCL_LOG_ABORT=4, NCCL_LOG_TRACE=5} ncclDebugLogLevel;
typedef enum {NCCL_INIT=1, NCCL_COLL=2, NCCL_P2P=4, NCCL_SHM=8, NCCL_NET=16, NCCL_GRAPH=32, NCCL_TUNING=64, NCCL_ENV=128, NCCL_ALLOC=256, NCCL_ALL=~0} ncclDebugLogSubSys;

typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...) __attribute__ ((format (printf, 5, 6)));

#define WARN(...) ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) ncclDebugLog(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

// Propagate errors up
#define NCCLCHECK(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    /* Print the back trace*/ \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    return res; \
  } \
} while (0);

#define SYSCHECKVAL(call, name, retval) do { \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed : %s", strerror(errno)); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECKSYNC(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    INFO(NCCL_ALL,"Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)

typedef struct {
  char* name;     // Used mostly for logging.
  char* pciPath;  // Path to the PCI device in /sys.
  uint64_t guid;  // Unique identifier for the NIC chip. Important for
                  // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport; // NCCL_PTR_HOST or NCCL_PTR_HOST|NCCL_PTR_CUDA
  int speed;      // Port speed in Mbps.
  int port;       // Port number.
  int maxComms;   // Maximum number of comms we can create
}ncclNetProperties_v4_t;

typedef ncclNetProperties_v4_t ncclNetProperties_t;

/* Error type */
typedef enum { ncclSuccess                 =  0,
               ncclUnhandledCudaError      =  1,
               ncclSystemError             =  2,
               ncclInternalError           =  3,
               ncclInvalidArgument         =  4,
               ncclInvalidUsage            =  5,
               ncclNumResults              =  6 } ncclResult_t;

/* Common socket address storage structure for IPv4/IPv6 */
union socketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

/* Init functions */
static int ncclNetIfs = -1;
struct ncclSocketDev {
  union socketAddress addr;
  char devName[MAX_IF_NAME_SIZE];
  char* pciPath;
};
static struct ncclSocketDev ncclSocketDevs[MAX_IFS];

pthread_mutex_t ncclSocketLock = PTHREAD_MUTEX_INITIALIZER;

/* Format a string representation of a (union socketAddress *) socket address using getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
static inline const char *socketToString(union socketAddress *addr, char *buf) {
  if (buf == NULL || addr == NULL) return NULL;
  struct sockaddr *saddr = &addr->sa;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) { buf[0]='\0'; return buf; }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  (void) getnameinfo(saddr, sizeof(union socketAddress), host, NI_MAXHOST, service, NI_MAXSERV, NI_NUMERICHOST|NI_NUMERICSERV);
  sprintf(buf, "%s<%s>", host, service);
  return buf;
}

static ncclResult_t socketProgressOpt(int op, int fd, union socketAddress *addr, void* ptr, int size, int* offset, int block) {
  int bytes = 0;
  char* data = (char*)ptr;
  char line[SOCKET_NAME_MAXLEN+1];
  do {
    if (op == NCCL_SOCKET_RECV) bytes = recv(fd, data+(*offset), size-(*offset), block ? 0 : MSG_DONTWAIT);
    if (op == NCCL_SOCKET_SEND) bytes = send(fd, data+(*offset), size-(*offset), block ? 0 : MSG_DONTWAIT);
    if (op == NCCL_SOCKET_RECV && bytes == 0) {
      WARN("Net : Connection closed by remote peer %s", socketToString(addr, line));
      return ncclSystemError;
    }
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        WARN("Net : Call to recv from %s failed : %s", socketToString(addr, line), strerror(errno));
        return ncclSystemError;
      } else {
        bytes = 0;
      }
    }
    (*offset) += bytes;
  } while (bytes > 0 && (*offset) < size);
  return ncclSuccess;
}

static ncclResult_t socketProgress(int op, int fd, union socketAddress *addr, void* ptr, int size, int* offset) {
  return socketProgressOpt(op, fd, addr, ptr, size, offset, 0);
}

static ncclResult_t ncclSocketGetPciPath(char* devName, char** pciPath) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/net/%s/device", devName);
  // May return NULL if the file doesn't exist.
  *pciPath = realpath(devicePath, NULL);
  return ncclSuccess;
}

ncclResult_t ncclSocketInit(ncclDebugLogger_t logFunction) {
  INFO(NCCL_ALL, "ncclSocketInit");
  if (ncclNetIfs == -1) {
    pthread_mutex_lock(&ncclSocketLock);
    if (ncclNetIfs == -1) {
      char names[MAX_IF_NAME_SIZE*MAX_IFS];
      union socketAddress addrs[MAX_IFS];
      ncclNetIfs = findInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (ncclNetIfs <= 0) {
        WARN("NET/Socket : no interface found");
        return ncclInternalError;
      } else {
        #define MAX_LINE_LEN (2047)
        char line[MAX_LINE_LEN+1];
        char addrline[SOCKET_NAME_MAXLEN+1];
        line[0] = '\0';
        addrline[SOCKET_NAME_MAXLEN] = '\0';
        for (int i=0; i<ncclNetIfs; i++) {
          strcpy(ncclSocketDevs[i].devName, names+i*MAX_IF_NAME_SIZE);
          memcpy(&ncclSocketDevs[i].addr, addrs+i, sizeof(union socketAddress));
          NCCLCHECK(ncclSocketGetPciPath(ncclSocketDevs[i].devName, &ncclSocketDevs[i].pciPath));
          snprintf(line+strlen(line), MAX_LINE_LEN-strlen(line), " [%d]%s:%s", i, names+i*MAX_IF_NAME_SIZE,
              socketToString(&addrs[i], addrline));
        }
        line[MAX_LINE_LEN] = '\0';
        INFO(NCCL_INIT|NCCL_NET,"NET/Socket : Using%s", line);
      }
    }
    pthread_mutex_unlock(&ncclSocketLock);
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketDevices(int* ndev) {
  *ndev = ncclNetIfs;
  INFO(NCCL_ALL, "ncclSocketDevices ndev=%d", *ndev);
  return ncclSuccess;
}

static ncclResult_t ncclSocketGetSpeed(char* devName, int* speed) {
  *speed = 0;
  char speedPath[PATH_MAX];
  sprintf(speedPath, "/sys/class/net/%s/speed", devName);
  int fd = open(speedPath, O_RDONLY);
  if (fd != -1) {
    char speedStr[] = "        ";
    if (read(fd, speedStr, sizeof(speedStr)-1) > 0) {
      *speed = strtol(speedStr, NULL, 0);
    }
    close(fd);
  }
  if (*speed <= 0) {
    INFO(NCCL_NET, "Could not get speed from %s. Defaulting to 10 Gbps.", speedPath);
    *speed = 10000;
  }
  INFO(NCCL_ALL, "ncclSocketGetSpeed devName=%s, speed=%d", devName, *speed);
  return ncclSuccess;
}

ncclResult_t ncclSocketGetProperties(int dev, ncclNetProperties_t* props) {
  INFO(NCCL_ALL, "ncclSocketGetProperties name=%s, pciPath=%s, dev=%d", ncclSocketDevs[dev].devName, ncclSocketDevs[dev].pciPath, dev);
  props->name = ncclSocketDevs[dev].devName;
  props->pciPath = ncclSocketDevs[dev].pciPath;
  props->guid = dev;
  props->ptrSupport = NCCL_PTR_HOST;
  NCCLCHECK(ncclSocketGetSpeed(props->name, &props->speed));
  props->port = 0;
  props->maxComms = 65536;
  return ncclSuccess;
}

ncclResult_t GetSocketAddr(int dev, union socketAddress* addr) {
  if (dev >= ncclNetIfs) return ncclInternalError;
  memcpy(addr, &ncclSocketDevs[dev].addr, sizeof(*addr));
  return ncclSuccess;
}

/* Communication functions */

#define MAX_SOCKETS 64
#define MAX_THREADS 16
#define MAX_REQUESTS NCCL_NET_MAX_REQUESTS
#define MIN_CHUNKSIZE (64*1024)

NCCL_PARAM(SocketNsocksPerThread, "NSOCKS_PERTHREAD", -2);
NCCL_PARAM(SocketNthreads, "SOCKET_NTHREADS", -2);

struct ncclSocketHandle {
  union socketAddress connectAddr;
  int nSocks;
  int nThreads;
};

struct ncclSocketTask {
  int op;
  void* data;
  int size;
  int fd;
  union socketAddress *addr;
  int offset;
  int used;
  ncclResult_t result;
};

struct ncclSocketRequest {
  int op;
  void* data;
  int size;
  int ctrlFd;
  union socketAddress *addr;
  int offset;
  int used;
  struct ncclSocketComm* comm;
  struct ncclSocketTask* tasks[MAX_SOCKETS];
  int nSubs;
};

struct ncclSocketTaskQueue {
  int next;
  int len;
  struct ncclSocketTask* tasks;
};

enum threadState {start, stop};

struct ncclSocketThreadResources {
  struct ncclSocketTaskQueue threadTaskQueue;
  enum threadState state;
  struct ncclSocketComm* comm;
  pthread_mutex_t threadLock;
  pthread_cond_t  threadCond;
};

struct ncclSocketListenComm {
  int fd;
  int nSocks;
  int nThreads;
};

struct ncclSocketComm {
  int ctrlFd;
  union socketAddress addr;
  int fds[MAX_SOCKETS];
  int nSocks;
  int nThreads;
  int nextFd;
  struct ncclSocketRequest requests[MAX_REQUESTS];
  pthread_t helperThread[MAX_THREADS];
  struct ncclSocketThreadResources threadResources[MAX_THREADS];
};

void* persistentSocketThread(void *args_) {
  struct ncclSocketThreadResources* resource = (struct ncclSocketThreadResources*)args_;
  struct ncclSocketComm* comm = resource->comm;
  volatile enum threadState* state = &resource->state;
  struct ncclSocketTaskQueue* myQueue = &resource->threadTaskQueue;
  int nSocksPerThread = comm->nSocks / comm->nThreads;
  while (1) {
    int idle = 1;
    int mark = myQueue->next; // mark newest task seen
    for (int i=0; i<myQueue->len; i+=nSocksPerThread) {
      int repeat;
      do {
        repeat = 0;
        for (int j=0; j<nSocksPerThread; j++) {
          struct ncclSocketTask* r = myQueue->tasks+i+j;
          if (r != NULL && r->used == 1 && r->offset < r->size) {
            r->result = socketProgress(r->op, r->fd, r->addr, r->data, r->size, &r->offset);
            if (r->result != ncclSuccess) {
              WARN("NET/Socket : socket progress error");
              return NULL;
            }
            idle = 0;
            if (r->offset < r->size) repeat = 1;
          }
        }
      } while (repeat);
    }
    if (idle) {
      pthread_mutex_lock(&resource->threadLock);
      while (mark == myQueue->next && *state != stop) { // no new tasks, wait
        pthread_cond_wait(&resource->threadCond, &resource->threadLock);
      }
      pthread_mutex_unlock(&resource->threadLock);
    }
    if (*state == stop) return NULL;
  }
}

ncclResult_t ncclSocketGetNsockNthread(int dev, int* ns, int* nt) {
  int nSocksPerThread = ncclParamSocketNsocksPerThread();
  int nThreads = ncclParamSocketNthreads();
  if (nThreads > MAX_THREADS) {
    WARN("NET/Socket : NCCL_SOCKET_NTHREADS is greater than the maximum allowed, setting to %d", MAX_THREADS);
    nThreads = MAX_THREADS;
  }
  if (nThreads == -2 || nSocksPerThread == -2) {
    // Auto-detection
    int autoNt=0, autoNs=1; // By default, we only use the main thread and do not spawn extra threads
    char vendorPath[PATH_MAX];
    snprintf(vendorPath, PATH_MAX, "/sys/class/net/%s/device/vendor", ncclSocketDevs[dev].devName);
    char* rPath = realpath(vendorPath, NULL);
    int fd = open(rPath, O_RDONLY);
    free(rPath);
    if (fd == -1) {
      // Could not find device vendor. This is handled silently so
      // we don't want to print an INFO error.
      TRACE(NCCL_NET, "Open of %s failed : %s", vendorPath, strerror(errno));
      goto end;
    }
    char vendor[7];
    strncpy(vendor, "0x0000", 7);
    int len;
    SYSCHECKVAL(read(fd, vendor, 6), "read", len);
    SYSCHECK(close(fd), "close");
    if (strcmp(vendor, "0x1d0f") == 0) { // AWS
      autoNt = 2;
      autoNs = 8;
    } else if (strcmp(vendor, "0x1ae0") == 0) { // GCP
      autoNt = 4;
      autoNs = 1;
    }
end:
    if (nThreads == -2) nThreads = autoNt;
    if (nSocksPerThread == -2) nSocksPerThread = autoNs;
  }
  int nSocks = nSocksPerThread * nThreads;
  if (nSocks > MAX_SOCKETS) {
    nSocksPerThread = MAX_SOCKETS/nThreads;
    WARN("NET/Socket : the total number of sockets is greater than the maximum allowed, setting NCCL_NSOCKS_PERTHREAD to %d", nSocksPerThread);
    nSocks = nSocksPerThread * nThreads;
  }
  *ns = nSocks;
  *nt = nThreads;
  if (nSocks > 0) INFO(NCCL_INIT, "NET/Socket: Using %d threads and %d sockets per thread", nThreads, nSocksPerThread);
  return ncclSuccess;
}

ncclResult_t ncclSocketNewListenComm(struct ncclSocketListenComm** comm) {
  NCCLCHECK(ncclCalloc(comm, 1));
  (*comm)->fd = -1;
  return ncclSuccess;
}

ncclResult_t ncclSocketNewComm(struct ncclSocketComm** comm) {
  NCCLCHECK(ncclCalloc(comm, 1));
  (*comm)->ctrlFd = -1;
  for (int i=0; i < MAX_SOCKETS; i++) {
    (*comm)->fds[i] = -1;
  }
  (*comm)->nextFd = 0;
  return ncclSuccess;
}

ncclResult_t ncclSocketListen(int dev, void* opaqueHandle, void** listenComm) {
  if (dev < 0) { // data transfer socket is based on specified dev
    return ncclInternalError;
  }
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  static_assert(sizeof(struct ncclSocketHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclSocketHandle size too large");
  struct ncclSocketListenComm* comm;
  NCCLCHECK(ncclSocketNewListenComm(&comm));
  NCCLCHECK(GetSocketAddr(dev, &handle->connectAddr));
  NCCLCHECK(createListenSocket(&comm->fd, &handle->connectAddr));
  NCCLCHECK(ncclSocketGetNsockNthread(dev, &comm->nSocks, &comm->nThreads));
  INFO(NCCL_ALL, "ncclSocketListen dev=%d, nSocks=%d, nThreads=%d", dev, comm->nSocks, comm->nThreads);
  handle->nSocks = comm->nSocks;
  handle->nThreads = comm->nThreads;
  *listenComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclSocketConnect(int dev, void* opaqueHandle, void** sendComm) {
  if (dev < 0) { // data transfer socket is based on specified dev
    return ncclInternalError;
  }
  struct ncclSocketComm* comm;
  NCCLCHECK(ncclSocketNewComm(&comm));
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  comm->nSocks = handle->nSocks;
  comm->nThreads = handle->nThreads;
  INFO(NCCL_ALL, "ncclSocketConnect dev=%d", dev);
  for (int i=0; i<comm->nSocks+1; i++) {
    int tmpFd, offset=0;
    NCCLCHECK(connectAddress(&tmpFd, &handle->connectAddr));
    NCCLCHECK(socketWait(NCCL_SOCKET_SEND, tmpFd, &handle->connectAddr, &i, sizeof(int), &offset));
    if (i == comm->nSocks) comm->ctrlFd = tmpFd;
    else comm->fds[i] = tmpFd;
  }
  *sendComm = comm;
  comm->addr = handle->connectAddr;
  return ncclSuccess;
}

ncclResult_t ncclSocketAccept(void* listenComm, void** recvComm) {
  struct ncclSocketListenComm* lComm = (struct ncclSocketListenComm*)listenComm;
  struct ncclSocketComm* rComm;
  NCCLCHECK(ncclSocketNewComm(&rComm));
  rComm->nSocks = lComm->nSocks;
  rComm->nThreads = lComm->nThreads;
  INFO(NCCL_ALL, "ncclSocketAccept lComm->fd=%d", lComm->fd);
  for (int i=0; i<rComm->nSocks+1; i++) {
    int tmpFd, sendSockIdx, offset=0;
    socklen_t socklen = sizeof(union socketAddress);
    SYSCHECKVAL(accept(lComm->fd, &rComm->addr.sa, &socklen), "accept", tmpFd);
    NCCLCHECK(socketWait(NCCL_SOCKET_RECV, tmpFd, &rComm->addr, &sendSockIdx, sizeof(int), &offset));
    if (sendSockIdx == rComm->nSocks) rComm->ctrlFd = tmpFd;
    else rComm->fds[sendSockIdx] = tmpFd;
  }
  *recvComm = rComm;
  return ncclSuccess;
}

ncclResult_t ncclSocketGetRequest(struct ncclSocketComm* comm, int op, void* data, int size, struct ncclSocketRequest** req) {
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclSocketRequest* r = comm->requests+i;
    if (r->used == 0) {
      r->op = op;
      r->data = data;
      r->size = size;
      r->ctrlFd = comm->ctrlFd;
      r->addr = &comm->addr;
      r->used = 1;
      r->comm = comm;
      r->nSubs = 0;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/Socket : unable to allocate requests");
  return ncclInternalError;
}

ncclResult_t ncclSocketGetTask(struct ncclSocketComm* comm, int op, void* data, int size, struct ncclSocketTask** req) {
  int tid = comm->nextFd % comm->nThreads;
  struct ncclSocketThreadResources* res = comm->threadResources+tid;
  struct ncclSocketTaskQueue* queue = &res->threadTaskQueue;
  // create helper threads and prepare per-thread task queue
  if (queue->tasks == NULL) {
    // each request can be divided up to nSocks tasks, and
    // these tasks are distributed to nThreads threads,
    // we need to make sure each thread queue has enough slots for MAX_REQUESTS
    queue->len = MAX_REQUESTS * DIVUP(comm->nSocks, comm->nThreads);
    NCCLCHECK(ncclCalloc(&queue->tasks, queue->len));
    queue->next = 0;
    res->comm = comm;
    pthread_mutex_init(&res->threadLock, NULL);
    pthread_cond_init(&res->threadCond, NULL);
    pthread_create(comm->helperThread+tid, NULL, persistentSocketThread, res);
  }
  struct ncclSocketTask* r = queue->tasks+queue->next;
  if (r->used == 0) {
    r->op = op;
    r->data = data;
    r->size = size;
    r->fd = comm->fds[comm->nextFd];
    r->addr = &comm->addr;
    r->offset = 0;
    r->result = ncclSuccess;
    comm->nextFd = (comm->nextFd + 1) % comm->nSocks;
    r->used = 1;
    *req = r;
    pthread_mutex_lock(&res->threadLock);
    queue->next = (queue->next+1)%queue->len;
    res->state = start;
    pthread_cond_signal(&res->threadCond);
    pthread_mutex_unlock(&res->threadLock);
    return ncclSuccess;
  }
  WARN("NET/Socket : unable to allocate subtasks");
  return ncclInternalError;
}

ncclResult_t ncclSocketTest(void* request, int* done, int* size) {
  INFO(NCCL_ALL, "ncclSocketTest, request=%p", request);
  *done = 0;
  struct ncclSocketRequest *r = (struct ncclSocketRequest*)request;
  if (r == NULL) {
    WARN("NET/Socket : test called with NULL request");
    return ncclInternalError;
  }
  if (r->used == 1) { /* try to send/recv size */
    int data = r->size;
    int offset = 0;
    NCCLCHECK(socketProgress(r->op, r->ctrlFd, r->addr, &data, sizeof(int), &offset));

    if (offset == 0) return ncclSuccess; /* Not ready -- retry later */

    // Not sure we could ever receive less than 4 bytes, but just in case ...
    if (offset < sizeof(int)) NCCLCHECK(socketWait(r->op, r->ctrlFd, r->addr, &data, sizeof(int), &offset));

    // Check size is less or equal to the size provided by the user
    if (r->op == NCCL_SOCKET_RECV && data > r->size) {
      char line[SOCKET_NAME_MAXLEN+1];
      WARN("NET/Socket : peer %s message truncated : receiving %d bytes instead of %d", socketToString(r->addr, line), data, r->size);
      return ncclInternalError;
    }
    r->size = data;
    r->offset = 0;
    r->used = 2; // done exchanging size
    // divide into subtasks
    int chunkOffset = 0, i = 0;
    if (r->comm->nSocks > 0) {
      // each request can be divided up to nSocks tasks
      int taskSize = std::max(MIN_CHUNKSIZE, DIVUP(r->size, r->comm->nSocks));
      while (chunkOffset < r->size) {
        int chunkSize = std::min(taskSize, r->size-chunkOffset);
        NCCLCHECK(ncclSocketGetTask(r->comm, r->op, (char*)(r->data)+chunkOffset, chunkSize, r->tasks+i++));
        chunkOffset += chunkSize;
      }
    }
    r->nSubs = i;
  }
  if (r->used == 2) { // already exchanged size
    if (r->nSubs > 0) {
      int nCompleted = 0;
      for (int i=0; i<r->nSubs; i++) {
        struct ncclSocketTask* sub = r->tasks[i];
        if (sub->result != ncclSuccess) return sub->result;
        if (sub->offset == sub->size) nCompleted++;
      }
      if (nCompleted == r->nSubs) {
        if (size) *size = r->size;
        *done = 1;
        r->used = 0;
        for (int i=0; i<r->nSubs; i++) {
          struct ncclSocketTask* sub = r->tasks[i];
          sub->used = 0;
        }
      }
    } else { // progress request using main thread
      if (r->offset < r->size) {
        NCCLCHECK(socketProgress(r->op, r->ctrlFd, r->addr, r->data, r->size, &r->offset));
      }
      if (r->offset == r->size) {
        if (size) *size = r->size;
        *done = 1;
        r->used = 0;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketRegMr(void* comm, void* data, int size, int type, void** mhandle) {
  INFO(NCCL_ALL, "ncclSocketRegMr comm=%p, data=%p, size=%d, type=%d", comm, data, size, type);
  return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}
ncclResult_t ncclSocketDeregMr(void* comm, void* mhandle) { 
  INFO(NCCL_ALL, "ncclSocketDeregMr comm=%p", comm);
  return ncclSuccess;
}

ncclResult_t ncclSocketIsend(void* sendComm, void* data, int size, void* mhandle, void** request) {
  printf("printf ncclSocketIsend data=%p, size=%d\n", data, size);
  INFO(NCCL_ALL, "ncclSocketIsend data=%p, size=%d", data, size);
  struct ncclSocketComm* comm = (struct ncclSocketComm*)sendComm;
  NCCLCHECK(ncclSocketGetRequest(comm, NCCL_SOCKET_SEND, data, size, (struct ncclSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclSocketIrecv(void* recvComm, void* data, int size, void* mhandle, void** request) {
  printf("printf ncclSocketIrecv data=%p, size=%d\n", data, size);
  INFO(NCCL_ALL, "ncclSocketIrecv data=%p, size=%d", data, size);
  struct ncclSocketComm* comm = (struct ncclSocketComm*)recvComm;
  NCCLCHECK(ncclSocketGetRequest(comm, NCCL_SOCKET_RECV, data, size, (struct ncclSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclSocketIflush(void* recvComm, void* data, int size, void* mhandle, void** request) {
  INFO(NCCL_ALL, "ncclSocketIflush data=%p, size=%d", data, size);
  // We don't support CUDA pointers, so we don't need a flush operation
  return ncclInternalError;
}

ncclResult_t ncclSocketCloseListen(void* opaqueComm) {
  INFO(NCCL_ALL, "ncclSocketCloseListen opaqueComm=%p", opaqueComm);
  struct ncclSocketListenComm* comm = (struct ncclSocketListenComm*)opaqueComm;
  if (comm) {
    if (comm->fd != -1) close(comm->fd);
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketClose(void* opaqueComm) {
  INFO(NCCL_ALL, "ncclSocketClose opaqueComm=%p", opaqueComm);
  struct ncclSocketComm* comm = (struct ncclSocketComm*)opaqueComm;
  if (comm) {
    for (int i=0; i<comm->nThreads; i++) {
      struct ncclSocketThreadResources* res = comm->threadResources+i;
      if (comm->helperThread[i]) {
        pthread_mutex_lock(&res->threadLock);
        res->state = stop;
        pthread_cond_signal(&res->threadCond);
        pthread_mutex_unlock(&res->threadLock);
        pthread_join(comm->helperThread[i], NULL);
      }
      free(res->threadTaskQueue.tasks);
    }
    if (comm->ctrlFd != -1) close(comm->ctrlFd);
    for (int i=0; i<comm->nSocks; i++) {
      if (comm->fds[i] != -1) close(comm->fds[i]);
    }
    free(comm);
  }
  return ncclSuccess;
}

ncclNet_t ncclNetSocket = {
  "BaguaNet",
  ncclSocketInit,
  ncclSocketDevices,
  ncclSocketGetProperties,
  ncclSocketListen,
  ncclSocketConnect,
  ncclSocketAccept,
  ncclSocketRegMr,
  ncclSocketDeregMr,
  ncclSocketIsend,
  ncclSocketIrecv,
  ncclSocketIflush,
  ncclSocketTest,
  ncclSocketClose,
  ncclSocketClose,
  ncclSocketCloseListen
};
