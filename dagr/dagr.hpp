/*

 Copyright (c) 2015-2018 Paul Keir, University of the West of Scotland.

 */

#ifndef _DAGR_HPP_
#define _DAGR_HPP_

#include <utility>          // std::forward
#ifdef TRISYCL
#include <CL/sycl.hpp>
#else
#include <SYCL/sycl.hpp>
#endif

template <typename T>
struct id { using type = T; };

// A type wrapper for write-only (+read-only), a trait, and a helper function
// Consider: forward<tag_t<X,char,'a'>::type&&>(x)
template <typename T, typename U, U tag>
struct tag_t {
  T &&t;
  using     type = T;
  using tag_type = U;
  static const U value{tag}; // unused?
};

namespace impl {

template <typename T, typename U, U tag> struct tag_match : std::false_type {};
template <typename T, typename U, U tag>
struct tag_match<tag_t<T,U,tag>,U,tag>                    : std::true_type  {};

template <typename>                      struct is_tag    : std::false_type {};
template <typename T, typename U, U tag>
struct is_tag<tag_t<T,U,tag>>                             : std::true_type  {};

template <typename T> struct detag                        : id<T>           {};
template <typename T, typename U, U tag>
struct detag<tag_t<T,U,tag>>                              : id<T>           {};

} // namespace impl

template <typename T, typename U, U tag>
struct tag_match : impl::tag_match<typename std::decay<T>::type,U,tag> {};

template <typename T, typename U, U tag>
constexpr bool tag_match_v() { return tag_match<T,U,tag>::value; }

template <typename T>
struct is_tag     : impl::is_tag<  typename std::decay<T>::type>       {};

template <typename T> constexpr bool is_tag_v() { return is_tag<T>::value; }

template <typename T>
struct detag : impl::detag<typename std::decay<T>::type> {};

template <typename T> using detag_t = typename detag<T>::type;

template <typename T, cl::sycl::access::mode Am>
using is_acc = tag_match<T,cl::sycl::access::mode,Am>;

// Write-only:
template <typename T> using is_wo = is_acc<T,cl::sycl::access::mode::write>;
template <typename T> constexpr bool is_wo_v() { return is_wo<T>::value; }

template <typename T>
using wo_tag_t = tag_t<T,cl::sycl::access::mode,cl::sycl::access::mode::write>;

// Read-only:
template <typename T> using is_ro = is_acc<T,cl::sycl::access::mode::read>;
template <typename T> constexpr bool is_ro_v() { return is_ro<T>::value; }

template <typename T>
using ro_tag_t = tag_t<T,cl::sycl::access::mode,cl::sycl::access::mode::read>;

namespace dagr {

template <typename T>
inline wo_tag_t<T> wo(T &&t) { return wo_tag_t<T>{std::forward<T>(t)}; }

template <typename T>
inline ro_tag_t<T> ro(T &&t) { return ro_tag_t<T>{std::forward<T>(t)}; }

} // namespace dagr

namespace static_asserts {

static_assert( is_wo_v<wo_tag_t<int>>(),"");
static_assert( is_wo_v<decltype(dagr::wo(42))>(),"");
static_assert( is_ro_v<ro_tag_t<int>>(),"");
static_assert( is_ro_v<decltype(dagr::ro(42))>(),"");
static_assert(!is_ro_v<wo_tag_t<int>>(),"");
static_assert(!is_wo_v<decltype(dagr::ro(42))>(),"");
static_assert(std::is_same<detag_t<      wo_tag_t<int>  >,int>::value,"");
static_assert(std::is_same<detag_t<const wo_tag_t<int> &>,int>::value,"");
static_assert(std::is_same<detag_t<               int   >,int>::value,"");
static_assert( is_tag_v<decltype(dagr::wo(3.14))>(),"");
static_assert( is_tag_v<decltype(dagr::ro(false))>(),"");
static_assert( is_tag_v<      tag_t<int,char,'a'>  >(),"");
static_assert( is_tag_v<const tag_t<int,char,'a'> &>(),"");
static_assert(!is_tag_v<int>(),"");
static_assert(!tag_match_v<int,char,'a'>(),"");
static_assert( tag_match_v<      tag_t<double,char,'a'>  ,char,'a'>(),"");
static_assert( tag_match_v<const tag_t<double,char,'a'> &,char,'a'>(),"");

} // namespace static_asserts

/* SYCL buffer trait */

namespace impl {

template <typename T>
struct is_buffer                          : std::false_type {};
template <typename T, int D, typename A>
struct is_buffer<cl::sycl::buffer<T,D,A>> : std::true_type {};

} // namespace impl

template <typename T>
using is_buffer = impl::is_buffer<typename std::decay<T>::type>;

template <typename T> constexpr bool is_buffer_v() {
  return is_buffer<T>::value;
}

namespace static_asserts {

static_assert(!is_buffer_v<                 float >(),"");
static_assert( is_buffer_v<cl::sycl::buffer<float>>(),"");

} // namespace static_asserts

/* SYCL accessor trait */

namespace impl {

using namespace cl::sycl;
template <typename T>
struct is_accessor                                : std::false_type {};
template <typename T, int D, access::mode Am, access::target At>
struct is_accessor<cl::sycl::accessor<T,D,Am,At>> : std::true_type {};

} // namespace impl

template <typename T>
using is_accessor = impl::is_accessor<typename std::decay<T>::type>;

template <typename T> constexpr bool is_accessor_v() {
  return is_accessor<T>::value;
}

// local_t - holds the arguments needed to allocate a local accessor.

template <typename T, int N>
struct local_t { using type = T; cl::sycl::range<N> r; };

namespace dagr {

template <typename T>
inline local_t<T,1> lo(const size_t x) {
  return local_t<T,1>{cl::sycl::range<1>{x}};
}

template <typename T>
inline local_t<T,2> lo(const size_t x, const size_t y) {
  return local_t<T,2>{cl::sycl::range<2>{x,y}};
}

template <typename T>
inline local_t<T,3> lo(const size_t x, const size_t y, const size_t z) {
  return local_t<T,3>{cl::sycl::range<3>{x,y,z}};
}

namespace impl {

template <typename T>
struct is_local               : std::false_type {};
template <typename T, int N>
struct is_local<local_t<T,N>> : std::true_type {};

} // namespace impl

template <typename T>
using is_local = impl::is_local<typename std::decay<T>::type>;

template <typename T> constexpr bool is_local_v() {
  return is_local<T>::value;
}

// At a later stage: kernel(&a[0],...) vs. kernel(a[0],...) i.e. ref vs. val
template <typename T>
constexpr bool pass_by_ref() { return is_buffer_v<T>() || is_local_v<T>(); }

namespace static_asserts {

static_assert(!is_buffer_v<                 float >(),"");
static_assert( is_buffer_v<cl::sycl::buffer<float>>(),"");

} // namespace static_asserts

} // namespace dagr

// template <bool ...Bs> using b_seq = std::integer_sequence<bool,Bs...>; // '14
template <bool ...>
  struct b_seq {};

template <cl::sycl::access::mode...>
  struct m_seq {};

template <bool ...Bs>    // 1... is only equal to cshift of 1... for 111111...
using all_true = std::is_same<b_seq<true, Bs...>, b_seq<Bs..., true>>;

template <bool ...Bs>
constexpr bool all_true_v() { return all_true<Bs...>::value; }

namespace static_asserts {
static_assert( all_true_v<true,true,true >(),"");
static_assert(!all_true_v<true,true,false>(),"");
} // namespace static_asserts

template <typename,unsigned>
  struct KernelId;

template <typename>
  struct get_item;

template <int N>
struct get_item<cl::sycl::nd_range<N>> { using type = cl::sycl::nd_item<N>; };

template <int N>
struct get_item<cl::sycl::range<N>>    { using type = cl::sycl::item<N>;    };

template <typename T>
using get_item_t = typename get_item<T>::type;

template <typename T>
constexpr
typename std::enable_if<!is_tag_v<T>(), cl::sycl::access::mode>::type
a_mode() {
  using namespace cl::sycl::access;
  using rr_t = typename std::remove_reference<T>::type;
  return std::is_const<rr_t>::value ? mode::read : mode::read_write;
}

template <typename T>       // write-only (tagged with wo_tag_t)
constexpr
typename std::enable_if< is_wo_v<T>(), cl::sycl::access::mode>::type
a_mode() {
  using rr_t = typename std::remove_reference<detag_t<T>>::type; // needed?
  static_assert(!std::is_const<rr_t>::value,
                "A const write-only buffer cannot exist.");
  return cl::sycl::access::mode::write;
}

template <typename T>       // read-only (tagged with ro_tag_t)
constexpr
typename std::enable_if< is_ro_v<T>(), cl::sycl::access::mode>::type
a_mode() {
  using rr_t = typename std::remove_reference<detag_t<T>>::type; // needed?
  return cl::sycl::access::mode::read;
}

template <cl::sycl::access::mode Am, typename T, int D, typename A>
inline
auto make_accessor(const cl::sycl::buffer<T,D,A> &x, cl::sycl::handler &cgh)
 -> decltype(
         const_cast<cl::sycl::buffer<T,D,A> &>(x).template get_access<Am>(cgh)){
  return const_cast<cl::sycl::buffer<T,D,A> &>(x).template get_access<Am>(cgh);
}

template <cl::sycl::access::mode Am, typename T, int N>
inline
auto make_accessor(const local_t<T,N> &l, cl::sycl::handler &cgh)
  -> decltype(cl::sycl::accessor<T,N,Am,
                                 cl::sycl::access::target::local>{l.r, cgh}) {
  return      cl::sycl::accessor<T,N,Am,
                                 cl::sycl::access::target::local>{l.r, cgh};
}

/*
// Using this instead of the acc struct, produces a compiler abort
template <bool B, typename A, typename = typename std::enable_if< B>::type>
inline auto acc_data(A a) -> decltype(&a[0]) { return &a[0]; }
template <bool B, typename A, typename = typename std::enable_if<!B>::type>
inline auto acc_data(A a) -> decltype( a[0]) { return  a[0]; }
*/
template <bool>
  struct acc;

template <>
struct acc<true> {
  template <typename A>
  #ifdef TRISYCL
  static auto data(A a) -> decltype(&a[0]) { return &a[0]; }
  #else // triSYCL doesn't have get_device_ptr and I don't believe it's in the
        //spec right now
   static auto data(A a) -> decltype(a.get_device_ptr()) {
       return a.get_device_ptr();
    }
  #endif
};

template <>
struct acc<false> {
  template <typename A>
  static auto data(A a) -> decltype( a[0]) { return a[0]; }
};

template <typename T, typename U, U tag>
inline auto remove_tag(const tag_t<T,U,tag> &o)
  -> decltype(std::forward<typename tag_t<T,U,tag>::type&&>(o.t)) {
  return      std::forward<typename tag_t<T,U,tag>::type&&>(o.t);
}
template <typename T>
inline const T &remove_tag(const T &x) { return x; }

template <typename T, int D, typename A>
inline const cl::sycl::buffer<T,D,A> &
make_buffer_if(const cl::sycl::buffer<T,D,A> &b) { return b; }

// Do nothing. No buffer-making here.
template <typename T, int N>
inline const local_t<T,N> &
make_buffer_if(const local_t<T,N> &l) { return l; }

template <typename T>
inline auto make_buffer_if(const T &x)
 -> decltype(cl::sycl::buffer<T,1>(const_cast<T*>(&x), cl::sycl::range<1>{1})) {
  return     cl::sycl::buffer<T,1>(const_cast<T*>(&x), cl::sycl::range<1>{1});
}

/*
Regarding on_stack's Xs value parameters: Non-buffer kernel arguments (x,xs...)
are now on the stack, so avoiding a bug. The buffer args too; it's fine;
although passed by value, and presumably copied, there must be reference
counting, as data isn't copied back until the original buffer is destroyed. One
the bug is gone, this should be removed.
*/

namespace dagr {
namespace impl {

using namespace cl::sycl;

template <typename K, unsigned Kn, typename R, typename Tm, typename Ta>
  struct dagr;

template <
  typename K,         // Kernel class which must contain a static method "k"
  unsigned Kn,        // Kernel name - (K,Kn) should be unique
  typename R,         // Range type - i.e. sycl::nd_range or sycl::range
  access::mode ...Ms, // Used when creating accessors
  bool ...Ps          // Identify which accessors should be passed by ref: &a
>
struct dagr<K,Kn,R,m_seq<Ms...>,b_seq<Ps...>> {

  template <typename ...As>
  static inline void run_acc(queue &q, const R r, handler &cgh, As ...as)
  {
    cgh.parallel_for<KernelId<K,Kn>>(r, [=](get_item_t<R> ix) {
      K::template k(ix,acc<Ps>::data(as)...);
    });
    static_assert(all_true_v<is_accessor_v<As>()...>(),
                  "All as must be accessors.");
  }

  template <typename ...Ts>
  static inline void run_buf(queue &q, const R r, const Ts &...bs) {
    q.submit([&](handler &cgh) {
      run_acc(q,r,cgh,make_accessor<Ms>(bs,cgh)...);
    });
//  static_assert(all_true_v<is_buffer_v<Ts>()...>(),"All bs must be buffers.");
//  some of bs now have type local_t
  }

  template <typename ...Xs>
  static inline void on_stack(queue &q, const R r, Xs ...xs) {
    run_buf(q,r,make_buffer_if(xs)...);
  }
};

} // namespace impl

// Add a template parameter to make the default mode for scalars (non-buffers)
// be read, rather than read_write. Consider discard_data
template <typename K, unsigned Kn, typename R, typename ...Xs>
void run(cl::sycl::queue q, const R r, Xs &&...xs)
{
  // static_assert that the number of Xs is equal to that of K::k
  // static_assert that K has a member named "k"
  static_assert(all_true_v<
                 !std::is_pointer<
                    typename std::remove_reference<detag_t<Xs>>::type
                  >::value...
                >(), "Pointer arguments to dagr::run are not permitted.");

  using are_buf_t = b_seq<pass_by_ref<detag_t<Xs>>()...>;
  using   modes_t = m_seq<a_mode<Xs>()...>;
  using    dagr_t = impl::dagr<K,Kn,R,modes_t,are_buf_t>;

  dagr_t::on_stack(q,r,remove_tag(xs)...);
}

} // namespace dagr

#endif // _DAGR_HPP_
