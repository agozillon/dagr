#include <cstring>
#include <SYCL/sycl.hpp>
#include "dagr.hpp"

struct abc {
  template <typename I, typename T>
  static void k(I ix, T *p, const char c) {
    p[ix.get_global_linear_id()] = c - 63 + ix.get_global_linear_id();
  }
};

int main(int argc, char *argv[])
{
  using namespace cl::sycl;
  char sz1 [65] = ""; sz1 [64] = '\0';
  char sz2 [65] = ""; sz2 [64] = '\0';
  const char expected[] =
    ";<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz";

  queue q;
  nd_range<3> ndr({4,4,4},{2,2,2});

  const char c = 'z';
  {
    buffer<char,1> buf(sz1, range<1>{64});
    dagr::run<abc,0>(q,ndr,buf,c);
  }
  printf("%s\n", sz1);

  dagr::run<abc,1>(q,ndr,buffer<char,1>(sz2,range<1>{64}),c);
  printf("%s\n", sz2);
  return strncmp(sz1,expected,64) || strncmp(sz2,expected,64);
}
