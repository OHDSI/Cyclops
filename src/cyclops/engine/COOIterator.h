#ifndef COOITERATOR_H_
#define COOITERATOR_H_


#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

#include <boost/iterator/iterator_facade.hpp>
#include <boost/tuple/tuple.hpp> 

//using namespace std;
using std::iterator_traits;
using std::get;
using std::ostream_iterator;
using std::cout;
using std::vector;
using std::endl;

template <class T0, class T1, class T2>
struct coo_helper_type {
  typedef boost::tuple<typename iterator_traits<T0>::value_type, 
  					   typename iterator_traits<T1>::value_type,
  					   typename iterator_traits<T2>::value_type
  					   > value_type;
  typedef boost::tuple<typename iterator_traits<T0>::value_type&, 
					   typename iterator_traits<T1>::value_type&,  
  					   typename iterator_traits<T2>::value_type&> ref_type;
  					   
  void swap(ref_type&& lhs, ref_type&& rhs) {
  	// Do nothing
  }
};

template <typename T0, typename T1, typename T2>
class coo_iterator : public boost::iterator_facade<coo_iterator<T0, T1, T2>,
                                                    typename coo_helper_type<T0, T1, T2>::value_type,
                                                    boost::random_access_traversal_tag,
                                                    typename coo_helper_type<T0, T1, T2>::ref_type> {
public:
   explicit coo_iterator(T0 iter0, T1 iter1, T2 iter2) : mIter0(iter0), mIter1(iter1), mIter2(iter2) {}
   typedef typename iterator_traits<T0>::difference_type difference_type;
private:
   void increment() { ++mIter0; ++mIter1; ++mIter2; }
   void decrement() { --mIter0; --mIter1; --mIter2; }
   bool equal(coo_iterator const& other) const { return mIter0 == other.mIter0; }
   typename coo_helper_type<T0, T1, T2>::ref_type dereference() const { 
		return (typename coo_helper_type<T0, T1, T2>::ref_type(*mIter0, *mIter1, *mIter2)); 
	}
   difference_type distance_to(coo_iterator const& other) const { return other.mIter0 - mIter0; }
   void advance(difference_type n) { mIter0 += n; mIter1 += n; mIter2 += n; }

   T0 mIter0;
   T1 mIter1;
   T2 mIter2;
   friend class boost::iterator_core_access;
};

template <typename T0, typename T1, typename T2>
coo_iterator<T0, T1, T2> make_iter(T0 t0, T1 t1, T2 t2) { return coo_iterator<T0, T1, T2>(t0, t1, t2); }

template <class T0, class T1, class T2> struct coo_iter_comp {
  typedef typename coo_helper_type<T0, T1, T2>::value_type T;
  bool operator()(const T& x1, const T& x2) {   		
  		return get<0>(x1) < get<0>(x2); 
  }
};

template <class T0, class T1, class T2> coo_iter_comp<T0, T1, T2> coo_make_comp(T0 t0, T1 t1, T2 t2) { return coo_iter_comp<T0, T1, T2>(); }


template <class T, class T2>
struct helper_type {
  typedef std::tuple<typename iterator_traits<T>::value_type, typename iterator_traits<T2>::value_type> value_type;
  typedef std::tuple<typename iterator_traits<T>::value_type&, typename iterator_traits<T2>::value_type&> ref_type;
  
  void swap(ref_type&& lhs, ref_type&& rhs) {
  	// Do nothing
  }  
};

template <typename T1, typename T2>
class dual_iterator : public boost::iterator_facade<dual_iterator<T1, T2>,
                                                    typename helper_type<T1, T2>::value_type,
                                                    boost::random_access_traversal_tag,
                                                    typename helper_type<T1, T2>::ref_type> {
public:
   explicit dual_iterator(T1 iter1, T2 iter2) : mIter1(iter1), mIter2(iter2) {}
   typedef typename iterator_traits<T1>::difference_type difference_type;
private:
   void increment() { ++mIter1; ++mIter2; }
   void decrement() { --mIter1; --mIter2; }
   bool equal(dual_iterator const& other) const { return mIter1 == other.mIter1; }
   typename helper_type<T1, T2>::ref_type dereference() const { return (typename helper_type<T1, T2>::ref_type(*mIter1, *mIter2)); }
   difference_type distance_to(dual_iterator const& other) const { return other.mIter1 - mIter1; }
   void advance(difference_type n) { mIter1 += n; mIter2 += n; }

   T1 mIter1;
   T2 mIter2;
   friend class boost::iterator_core_access;
};

template <typename T1, typename T2>
dual_iterator<T1, T2> make_iter(T1 t1, T2 t2) { return dual_iterator<T1, T2>(t1, t2); }

template <class T1, class T2> struct iter_comp {
  typedef typename helper_type<T1, T2>::value_type T;
  bool operator()(const T& t1, const T& t2) { return get<0>(t1) < get<0>(t2); }
};

// template <typename T1, typename T2>
// void swap(typename helper_type<T1,T2>::ref_type&& lhs, typename helper_type<T1,T2>::ref_type&& rhs) {
// 	// Do nothing
// }

namespace std {

void swap(std::tuple<double&, char&> lhs, std::tuple<double&, char&> rhs) {
	std::swap(get<0>(lhs), get<0>(rhs));
	std::swap(get<1>(lhs), get<1>(rhs));	
}

void swap(std::tuple<int&, int&> lhs, std::tuple<int&, int&> rhs) {
	std::swap(get<0>(lhs), get<0>(rhs));
	std::swap(get<1>(lhs), get<1>(rhs));	
}

}

template <class T1, class T2> iter_comp<T1, T2> make_comp(T1 t1, T2 t2) { return iter_comp<T1, T2>(); }

// template<class T> void print(T& items) {
//   copy(items.begin(), items.end(), ostream_iterator<typename T::value_type>(cout, " ")); cout << endl;
// }

// int main() {
//   vector<double> nums1 = {3, 2, 1, 0};
//   vector<char> nums2 = {'D','C', 'B', 'A'};
//   sort(make_iter(nums1.begin(), nums2.begin()), 
//        make_iter(nums1.end(), nums2.end()), 
//        make_comp(nums1.begin(), nums2.begin()));
//   print(nums1);
//   print(nums2);
// }



template<class T> void print(T& items) {
  copy(items.begin(), items.end(), ostream_iterator<typename T::value_type>(cout, " ")); cout << endl;
}

// int main() {
//   vector<double> nums1 = {3, 2, 1, 0};
//   vector<char> nums2 = {'D','C', 'B', 'A'};
//   sort(make_iter(nums1.begin(), nums2.begin()), 
//        make_iter(nums1.end(), nums2.end()), 
//        make_comp(nums1.begin(), nums2.begin()));
//   print(nums1);
//   print(nums2);
// }


#endif // COOITERATOR_H_