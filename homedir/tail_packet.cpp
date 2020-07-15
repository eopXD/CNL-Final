#include <cstddef>
#include <cstdint>

class File {
 public:
  File() : fd_(-1) {}
  File(const char* filename, bool create = true) : fd_(-1) {
    open(filename, create);
  }

  // STL container- / fstream-like names
  size_t size() const;
  bool resize(size_t sz);
  bool clear() { return resize(0); }
  void close();
  void open(const char* filename, bool create = true);
  bool is_open() const { return fd_ != -1; }
  int get_fd() const { return fd_; }
  bool read(size_t offset, uint8_t* res, size_t sz) const;
  bool write(size_t offset, const uint8_t* res, size_t sz);

 private:
  int fd_;
};

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

inline size_t File::size() const {
  if (!is_open()) return 0;
  struct stat st;
  if (fstat(fd_, &st) < 0) return 0;
  return st.st_size;
}

inline bool File::resize(size_t sz) {
  if (!is_open()) return false;
  return ftruncate(fd_, sz) >= 0;
}

inline void File::close() { // clear cache if cache is implemented
  if (fd_ >= 0) ::close(fd_), fd_ = -1;
}

inline void File::open(const char* filename, bool create) {
  close();
  fd_ = ::open(filename, create ? O_RDWR | O_CREAT : O_RDWR, 0600);
}

inline bool File::read(size_t offset, uint8_t* res, size_t sz) const {
  return pread(fd_, res, sz, offset) == (ssize_t)sz;
}

inline bool File::write(size_t offset, const uint8_t* res, size_t sz) {
  return pwrite(fd_, res, sz, offset) == (ssize_t)sz;
}

// --------------------------------
#include <string>
#include <vector>
#include <cstring>
#include <algorithm>

struct Row {
  long long tm;
  int rssi;
  char mac[13];
};

long long FindPrevLine(File& f, long long a) {
  if (a <= 0) return 0;
  static char buf[256];
  for (long long i = a + 1;; i -= sizeof(buf)) {
    long long start = std::max(i - (long long)sizeof(buf), 0ll);
    size_t n = i - start;
    f.read(start, (uint8_t*)buf, n);
    for (size_t j = n - 1; j >= 0; j--) {
      if (buf[j] == '\n') return start + j + 1;
    }
    if (i == 0) return 0;
  }
}

long long GetTime(File& f, long long start) {
  char buf[22];
  f.read(start, (uint8_t*)buf, sizeof(buf));
  if (buf[19] != ',') return -1;
  try {
    return std::stoll(std::string(buf, buf + 19));
  } catch (...) {
    return -1;
  }
}

std::vector<Row> Read(File& f, long long start, long long end = -1) {
  static char buf[8192];
  if (end == -1) end = f.size();

  std::vector<Row> ret;
  long long tm = 0, tm2 = 0;
  char header[45];
  int rssi = 0, freq = 0, len = 0, szh = 0, szf = 0, stage = 0, num = 0;
  auto Reset = [&](){tm = tm2 = rssi = freq = len = stage = szh = szf = num = 0;};
  auto AppendInt = [&](auto& x, char c, int lim) {
    if (c == ',') {
      stage++, num = 0;
    } else if (c < '0' || c > '9') {
      stage = -1;
    } else {
      x = x * 10 + c - '0';
      if (num++ >= lim) stage = -1;
    }
  };
  for (long long i = start; i < end; i += sizeof(buf)) {
    size_t n = std::min((size_t)(end - i), sizeof(buf));
    f.read(i, (uint8_t*)buf, n);
    for (size_t j = 0; j < n; j++) {
      switch (stage) {
        case -1:
          if (buf[j] == '\n') Reset();
          break;
        case 0:
          AppendInt(tm, buf[j], 19);
          if (stage == 1 && tm < 1'000'000'000'000'000'000) stage = -1;
          break;
        case 1:
          AppendInt(tm2, buf[j], 19);
          break;
        case 2:
          if (buf[j] == ',') {
            header[szh++] = '\0', stage++;
          } else if ((buf[j] < '0' || buf[j] > '9') && (buf[j] < 'a' || buf[j] > 'f')) {
            stage = -1;
          } else {
            header[szh++] = buf[j];
            if (szf >= 44) stage = -1;
          }
          break;
        case 3:
          if (buf[j] == '-') {
            if (rssi != 0) stage = -1;
          } else if (buf[j] == ',') {
            if (rssi == -256) stage = -1;
            else stage++;
          } else if (buf[j] < '0' || buf[j] > '9') {
            stage = -1;
          } else {
            rssi = rssi * 10 - (buf[j] - '0');
          }
          break;
        case 4:
          AppendInt(freq, buf[j], 4);
          if (stage != 4 && freq != 2442) stage = -1;
          break;
        case 5:
          AppendInt(len, buf[j], 6);
          break;
        case 6:
          if (buf[j] == '\n') {
            if (szf == 64 &&
                ((header[0] == '0' && header[1] == '8') ||
                 (header[0] == '1' && header[1] == '1') ||
                 (header[0] == '2' && header[1] == '0') ||
                 (header[0] == '5' && header[1] == '0') ||
                 (header[0] == '5' && header[1] == '4') ||
                 (header[0] == '8' && header[1] == '0') ||
                 (header[0] == '8' && header[1] == '4') ||
                 (header[0] == '8' && header[1] == '8') ||
                 (header[0] == '9' && header[1] == '4') ||
                 (header[0] == 'a' && header[1] == '4') ||
                 (header[0] == 'b' && header[1] == '0') ||
                 (header[0] == 'b' && header[1] == '4') ||
                 (header[0] == 'd' && header[1] == '0'))) {
              ret.emplace_back();
              ret.back().tm = tm;
              ret.back().rssi = rssi;
              memcpy(ret.back().mac, header + 20, 12);
              ret.back().mac[12] = '\0';
            }
            Reset();
          } else {
            if ((buf[j] < '0' || buf[j] > '9') && (buf[j] < 'a' || buf[j] > 'f')) {
              stage = -1;
            } else {
              if (szf++ >= 64) stage = -1;
            }
          }
          break;
        default: throw;
      }
    }
  }
  return ret;
}

int main(int argc, char** argv) {
  if (argc < 3) return 1;
  long long sec = std::stoll(argv[1]);
  File f(argv[2]);
  long long tm = 0;
  long long sz = f.size();
  while (true) {
    long long p = FindPrevLine(f, sz - 2);
    tm = GetTime(f, p);
    if (tm >= 1'000'000'000'000'000'000 && tm <= 3'000'000'000'000'000'000) break;
    sz = p;
  }
  long long offset = 4096;
  for (; offset < sz; offset = offset * 3 / 2) {
    if (GetTime(f, FindPrevLine(f, sz - offset)) < tm - sec * 1'000'000'000) break;
  }
  if (offset >= sz) offset = sz;
  long long v = FindPrevLine(f, sz - offset);
  auto ret = Read(f, v);
  for (auto& r : ret) printf("%lld,%d,0,0,%s\n", r.tm, r.rssi, r.mac);
}
