#include <cstring>
#include <SYCL/sycl.hpp>
#include "dagr.hpp"

// Like bugs/local_barrier_bug.cpp, this barrier code needs -O2 or -O3 flags
// Instead of lo<char>(8), a 64 char device-side buffer should also work.

struct abc_local {

  template <typename T, typename U>
  static void k(cl::sycl::nd_item<1> ix, T *p, U *p_local)
  {
    const auto extent   = ix.get_local_range(0) - 1;
    const auto local_id = ix.get_local()[0];

    p_local[local_id]     = 'z' - 63 + ix.get_global()[0];
    ix.barrier(cl::sycl::access::fence_space::local_space);
    p[ix.get_global()[0]] = p_local[extent - local_id];
  }

};

int main(int argc, char *argv[])
{
  using namespace cl::sycl;
  char sz[65] = ""; sz[64] = '\0';
  const char expected[] =                               // ...stuvwxyz
    "BA@?>=<;JIHGFEDCRQPONMLKZYXWVUTSba`_^]\\[jihgfedcrqponmlkzyxwvuts";

  queue q;
  nd_range<1> ndr(range<1>{64},range<1>{8});

  {
    using dagr::lo;        // local memory
    buffer<char,1> data(sz,range<1>{64});
    dagr::run<abc_local,0>(q,ndr,data,lo<char>(8));
  }

  printf("%s\n", sz);
  return strncmp(sz,expected,64);
}
