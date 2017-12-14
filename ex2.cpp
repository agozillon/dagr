#include <SYCL/sycl.hpp>
#include "dagr.hpp"

struct vec_add {
  template <typename I, typename T>
  static void k(I ix, const T *a, const T *b, T *c) {
    c[ix[0]] = a[ix[0]] + b[ix[0]];
  }
};

inline float vec_elem_sum(float f)            { return f; }
//inline float vec_elem_sum(cl::sycl::float3 f) { return f.x() + f.y() + f.z(); }
inline float vec_elem_sum(cl::sycl::float3 f) { return f.x() + (f.y() + f.z()); }

template <unsigned N, typename T>
bool go(cl::sycl::queue &q, T *data_a, T *data_b, T *data_c,
        const unsigned sz, const float expected_total)           {
  using namespace cl::sycl;

  const auto r = range<1>(sz);
  const buffer<T,1> buf_a(data_a, r);
  const buffer<T,1> buf_b(data_b, r);

  {
    buffer<T,1> buf_c(data_c, r);
    dagr::run<vec_add,N>(q,r,buf_a,buf_b,buf_c);
  }

  float total{0};
  for (unsigned i = 0; i < sz; i++)
    total += vec_elem_sum(data_c[i]);

  bool res = (total == expected_total);
  printf("         total:%3.1f\n", total);
  printf("expected total:%3.1f\n", expected_total);
  printf("%s\n", res ? "OK" : "ERROR");

  return res;
}

int main(int argc, char *argv[])
{
  using namespace cl::sycl;
  const unsigned sz = 64;
  float  data_a [sz]; float  data_b [sz]; float  data_c [sz];
  float3 data_a3[sz]; float3 data_b3[sz]; float3 data_c3[sz];

  for (unsigned i = 0; i < sz; i++) {
    data_a [i] = i;
    data_b [i] = i;
    data_a3[i].x() = i; data_a3[i].y() = i; data_a3[i].z() = i;
    data_b3[i].x() = i; data_b3[i].y() = i; data_b3[i].z() = i;
  }

  queue q;
  bool res = true; 
  res &= go<0>(q,data_a, data_b, data_c, sz,  static_cast<float>(sz*(sz-1)));
//  res &= go<1>(q,data_a3,data_b3,data_c3,sz,3*static_cast<float>(sz*(sz-1)));

  return res ? 0 : -1;
}
