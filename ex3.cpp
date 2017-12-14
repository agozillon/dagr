#include <cstring>
#include <SYCL/sycl.hpp>
#include "dagr.hpp"

struct abc_tmp {
  template <typename I, typename T>
  static void k(const I ix, T *p_tmp, const char c) {
    p_tmp[ix.get_linear_id()] = c - 63 + ix.get_linear_id();
  }
};

struct abc {
  template <typename I, typename T>
  static void k(const I ix, T *p, const T *p_tmp, const char c) {
    p[ix.get_linear_id()] = p_tmp[ix.get_linear_id()];
  }
};

int main(int argc, char *argv[])
{
  using namespace cl::sycl;
  int ret = 0;
  const char c = 'z';
  char sz[65] = ""; sz[64] = '\0';
  const char expected[] =
    ";<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz";
  queue q;
  auto r = range<3>{4,4,4};

  for (int run = 0; run < 3; ++run)
  {
    {
      using dagr::wo;  // write-only
      using dagr::ro;  // read-only
      buffer<char,1> buf_tmp(   range<1>{64}); // Allocated on the device only
      buffer<char,1> buf    (sz,range<1>{64});

           if (0 == run)                  // tagged lvalues
      {
        dagr::run<abc_tmp,0>(q,r,        wo(buf_tmp),c);
        dagr::run<abc    ,0>(q,r,wo(buf),ro(buf_tmp),c);
      } else if (1 == run) {              // wo-tagged rvalue
        dagr::run<abc_tmp,0>(q,r,        wo(buf_tmp),c);
        dagr::run<abc    ,0>(q,r,wo(buffer<char,1>(sz,range<1>{64})),
                                         ro(buf_tmp),c);
      } else if (2 == run) {              // tagged lvalues
        auto wo_buf_tmp = wo(buf_tmp);
        auto ro_buf_tmp = ro(buf_tmp);
        auto wo_buf     = wo(buf);
        dagr::run<abc_tmp,0>(q,r,       wo_buf_tmp,c);
        dagr::run<abc    ,0>(q,r,wo_buf,ro_buf_tmp,c);
      }
    } // sycl::buffer scoped lifetime
    printf("%s\n", sz);
    ret |= strncmp(sz,expected,64);
  }

  return ret;
}

