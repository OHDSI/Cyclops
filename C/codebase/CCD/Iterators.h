#ifndef ITERATORS_H
#define ITERATORS_H

#include "CompressedIndicatorMatrix.h"

// Define some constants
const unsigned int RowMajorBit = 0x1;

// Generic iterator for a dense column
// template<typename Derived> 
// class DenseBase<Derived>::InnerIterator
// {
//   protected:
//     typedef typename Derived::Scalar Scalar;
//     typedef typename Derived::Index Index;
// 
//     enum { IsRowMajor = (Derived::Flags&RowMajorBit)==RowMajorBit };
//     
//   public:
//     EIGEN_STRONG_INLINE InnerIterator(const Derived& expr, Index outer)
//       : m_expression(expr), m_inner(0), m_outer(outer), m_end(expr.innerSize())
//     {}
// 
//     EIGEN_STRONG_INLINE Scalar value() const
//     {
//       return (IsRowMajor) ? m_expression.coeff(m_outer, m_inner)
//                           : m_expression.coeff(m_inner, m_outer);
//     }
// 
//     EIGEN_STRONG_INLINE InnerIterator& operator++() { m_inner++; return *this; }
// 
//     EIGEN_STRONG_INLINE Index index() const { return m_inner; }
//     inline Index row() const { return IsRowMajor ? m_outer : index(); }
//     inline Index col() const { return IsRowMajor ? index() : m_outer; }
// 
//     EIGEN_STRONG_INLINE operator bool() const { return m_inner < m_end && m_inner>=0; }
// 
//   protected:
//     const Derived& m_expression;
//     Index m_inner;
//     const Index m_outer;
//     const Index m_end;
// };

// Generic iterator for a sparse column
//template<typename Scalar, typename Index>
class SparseIndicatorIterator {
  public:

	typedef real Scalar;
	typedef int Index;

	inline SparseIndicatorIterator(const CompressedIndicatorMatrix& mat, Index column)
	  : //mValues(NULL),
	    mIndices(mat.getCompressedColumnVector(column)),
//	    mOuter(column),
	    mId(0), mEnd(mat.getNumberOfEntries(column)){
		// Do nothing
	}

//    InnerIterator(const SparseMatrix& mat, Index outer)
//      : mValues(mat._valuePtr()), mIndices(mat._innerIndexPtr()),
//        mOuter(outer), mId(mat.m_outerIndex[outer]), mEnd(mat.m_outerIndex[outer+1])
//    {}

//    enum { IsRowMajor = (RowMajor & RowMajorBit) == RowMajorBit };

    inline SparseIndicatorIterator& operator++() { ++mId; return *this; }

//    inline const Scalar& value() const { return mValues[mId]; }
//    inline Scalar& valueRef() { return const_cast<Scalar&>(mValues[mId]); }

    inline const Scalar value() const { return static_cast<Scalar>(1); }
//    inline Scalar& value() const { return ref; }

    inline Index index() const { return mIndices[mId]; }
//    inline Index outer() const { return mOuter; }
//    inline Index row() const { return IsRowMajor ? mOuter : index(); }
//    inline Index col() const { return IsRowMajor ? index() : mOuter; }

    inline operator bool() const { return (mId < mEnd); }

  protected:
//    const Scalar* mValues;
    const Index* mIndices;
//    const Index mOuter;
    Index mId;
    const Index mEnd;
};

#endif // ITERATORS_H
