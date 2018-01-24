#include <SYCL/sycl.hpp>
#include "dagr/dagr.hpp"

// This tests calls to OpenCL builtins: clamp and normalize

struct cl_calls {
  template <typename I, typename T, typename U>
  static void k(I ix, T *out1, U *out2, const T *in) {
    cl::sycl::float3 f3{10.0f,20.0f,30.0f};
    out1[ix[0]] = cl::sycl::clamp(in[ix[0]], 5.0f, 10.0f);
    out2[ix[0]] = cl::sycl::normalize(f3);
  }
};

int main(int argc, char *argv[])
{
  using namespace cl::sycl;

  const  unsigned sz = 16;
  float  data_in[sz] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  float  data_out1[sz];
  float3 data_out2[sz];
  const auto r = range<1>(sz);

  queue q; // default ok with maths calls?
  //queue q(intel_selector{});

  {
    buffer<float,1>  buf_in(data_in, r);
    buffer<float,1>  buf_out1(data_out1, r);
    buffer<float3,1> buf_out2(data_out2, r);
    dagr::run<cl_calls,0>(q,r,buf_out1,buf_out2,buf_in);
  }

  for (int i = 0; i < sz; i++) {
    auto print = [](float s, float x, float y, float z) {
      printf("%g (%.3g,%.3g,%.3g)\n", s, x, y, z);
    };
    printf("a: ");
    print(data_out1[i],data_out2[i].x(),data_out2[i].y(),data_out2[i].z());
  }

  return 0;
}
