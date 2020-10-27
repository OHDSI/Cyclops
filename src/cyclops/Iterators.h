#ifndef ITERATORS_H
#define ITERATORS_H

#include <boost/tuple/tuple.hpp>

#include "CompressedDataMatrix.h"

namespace bsccs {

/**
 * Iterators for dense, sparse and indicator vectors.  Each can be passed as a
 * template parameter to functions for compile-time specialization and optimization.
 * If performance is not an issue, GenericIterator wraps all three formats into a
 * single run-time determined iterator
 */


struct IndicatorTag {};
struct SparseTag {};
struct DenseTag {};
struct InterceptTag {};

template <typename Scalar>
class ReverseIndicatorIterator;

// Iterator for a sparse of indicators column
template <typename Scalar>
class IndicatorIterator {
  public:

	typedef IndicatorTag tag;
	typedef int Index;
	typedef Scalar ValueType;
	typedef boost::tuples::tuple<Index> XTuple;

	const static std::string name;

	static const bool isIndicatorStatic = true;
	enum  { isIndicator = true };
	enum  { isSparse = true };

// 	inline IndicatorIterator(const CompressedDataColumn& col)
// 	  : mIndices(col.getColumns()),
// 	    mId(0), mEnd(

	inline IndicatorIterator(const CompressedDataMatrix<Scalar>& mat, Index column)
	  : mIndices(mat.getCompressedColumnVector(column)),
	    mId(0), mEnd(mat.getNumberOfEntries(column)){
		// Do nothing
	}

	inline IndicatorIterator(const std::vector<int>& vec, Index max = 0)
	: mIndices(vec.data()), mId(0), mEnd(vec.size()) {
		// Do nothing
	}

	inline IndicatorIterator(const std::vector<int>* vec, Index max = 0)
	    : mIndices(vec->data()), mId(0), mEnd(vec->size()) {
	    // Do nothing
	}

//	inline IndicatorIterator(const std::vector<int>& vec, Index column, Index max)
//	: mIndices(vec.data()), mId(0), mEnd(vec.size()) {
//		// Do nothing
//	}

    inline IndicatorIterator& operator++() { ++mId; return *this; }
	inline IndicatorIterator operator++(int) { auto rtn = *this; ++(*this); return rtn; }

    inline const Scalar value() const { return static_cast<Scalar>(1); }

    inline Index index() const { return mIndices[mId]; }
	inline Index nextIndex() const { return mIndices[mId + 1]; }
    inline operator bool() const { return (mId < mEnd); }
	inline bool inRange(const Scalar i) const { return (mId < i); }
    inline Index size() const { return mEnd; }
	inline Scalar multiply(const Scalar x) const { return x; }
	inline Scalar multiply(const Scalar x, const Index index) { return x; }

	inline const Index* getIndices() const { return mIndices; }

	ReverseIndicatorIterator<Scalar> reverse() const {
	    return ReverseIndicatorIterator<Scalar>(*this);
	}

  protected:
    const Index* mIndices;
    Index mId;
    const Index mEnd;
};

template <typename Scalar>
class ReverseIndicatorIterator {
public:

    typedef int Index;

    inline ReverseIndicatorIterator(const IndicatorIterator<Scalar>& forward)
        : mIndices(forward.getIndices()), mEnd(forward.size()), mId(mEnd - 1) {}

    inline ReverseIndicatorIterator& operator--() { --mId; return *this; }
    inline operator bool() const { return (mId >= 0); }

    inline const Scalar value() const { return static_cast<Scalar>(1); }
    inline Index index() const { return mIndices[mId]; }

    inline Index size() const { return mEnd; }

protected:
    const Index* mIndices;
    const Index mEnd;
    Index mId;
};

template <typename Scalar>
class ReverseSparseIterator;

// Iterator for a sparse column
template <typename Scalar>
class SparseIterator {
  public:

	typedef SparseTag tag;
	typedef int Index;
	typedef Scalar ValueType;
	typedef boost::tuples::tuple<Index, Scalar> XTuple;

	const static std::string name;

	static const bool isIndicatorStatic = false;
	enum  { isIndicator = false };
	enum  { isSparse = true };

	inline SparseIterator(const CompressedDataMatrix<Scalar>& mat, Index column)
	  : mValues(mat.getDataVector(column)), mIndices(mat.getCompressedColumnVector(column)),
	    mId(0), mEnd(mat.getNumberOfEntries(column)){
		// Do nothing
	}

	inline SparseIterator(const CompressedDataColumn<Scalar>& column)
	  : mValues(column.getData()), mIndices(column.getColumns()),
	    mId(0), mEnd(column.getNumberOfEntries()){
		// Do nothing
	}

	inline SparseIterator(const std::vector<int>& vec, Index max = 0)
	: mIndices(vec.data()), mId(0), mEnd(vec.size()) {
		// Do nothing
	}

	inline SparseIterator(const std::vector<int>* vec, Index max = 0)
	    : mIndices(vec->data()), mId(0), mEnd(vec->size()) {
	    // Do nothing
	}

    inline SparseIterator& operator++() { ++mId; return *this; }
	inline SparseIterator operator++(int) { auto rtn = *this; ++(*this); return rtn; }


//	bool operator==(iterator other) const {return num == other.num;}
//	bool operator!=(iterator other) const {return !(*this == other);}

    inline const Scalar value() const {
    	return mValues[mId]; }
    inline Scalar& valueRef() { return const_cast<Scalar&>(mValues[mId]); }

    inline Index index() const { return mIndices[mId]; }
	inline Index nextIndex() const { return mIndices[mId + 1]; }
    inline operator bool() const { return (mId < mEnd); }
	inline bool inRange(const Scalar i) const { return (mId < i); }
    inline Index size() const { return mEnd; }
	inline Scalar multiply(const Scalar x) const { return x * mValues[mId]; }
	inline Scalar multiply(const Scalar x, const Index index) const { return x * mValues[index]; }

	inline const Scalar* getValues() const { return mValues; }
	inline const Index* getIndices() const { return mIndices; }

	ReverseSparseIterator<Scalar> reverse() const {
	    return ReverseSparseIterator<Scalar>(*this);
	}

  protected:
    const Scalar* mValues;
    const Index* mIndices;
    Index mId;
    const Index mEnd;
};

template <typename Scalar>
class ReverseSparseIterator {
public:

    typedef int Index;

    inline ReverseSparseIterator(const SparseIterator<Scalar>& forward)
        : mValues(forward.getValues()), mIndices(forward.getIndices()),
        mEnd(forward.size()), mId(mEnd - 1) {}

    inline ReverseSparseIterator& operator--() { --mId; return *this; }
    inline operator bool() const { return (mId >= 0); }

    inline const Scalar value() const { return mValues[mId]; }
    inline Index index() const { return mIndices[mId]; }

    inline Index size() const { return mEnd; }

protected:
    const Scalar* mValues;
    const Index* mIndices;
    const Index mEnd;
    Index mId;
};

template <typename IteratorType, typename Scalar>
class DenseView {
public:
	typedef int Index;
    typedef Scalar ValueType;

	enum  { isIndicator = false };
	enum  { isSparse = false };

	inline DenseView(const IteratorType& iterator, Index start, Index end)
		: mIterator(iterator), mId(start), mEnd(end) {
		while (mIterator && mIterator.index() < start) {
			++mIterator;
		}
	}

    inline DenseView& operator++() {
    	if (mIterator.index() == mId) {
    		++mIterator;
    	}
    	++mId;
    	return *this;
    }

    inline const Scalar operator*() const {
    	return value();
    }

    inline const Scalar value() const {
    	if (mIterator.index() == mId) {
    		return mIterator.value();
    	} else {
    		return static_cast<Scalar>(0.0);
    	}
    }

    inline Index index() const { return mId; }
	inline Index nextIndex() const { return mId + 1; }
    inline operator bool() const { return (mId < mEnd); }
    inline Index size() const { return mEnd; }
	inline Scalar multiply(const Scalar x) const { return x * value(); }


protected:
	IteratorType mIterator;
	Index mId;
	const Index mEnd;
};

// Counting iterator
class CountingIterator {
public:
	typedef int Index;

//	static const bool isIndicator = false;
	enum  { isIndicator = false };

	inline CountingIterator(const std::vector<int>& vec, Index end)
	: mId(0), mEnd(end) {
		// Do nothing
	}

	inline CountingIterator(const std::vector<int>* vec, Index end)
	    : mId(0), mEnd(end) {
	    // Do nothing
	}

	inline CountingIterator& operator++() { ++mId; return *this; }
	inline Index index() const  { return mId; }
	inline Index nextIndex() const { return mId + 1; }
	inline operator bool() const { return (mId < mEnd); }
    inline Index size() const { return mEnd; }

protected:
	Index mId;
	const Index mEnd;
};

template <typename Scalar>
class ReverseDenseIterator;

// Iterator for a dense column
template <typename Scalar>
class DenseIterator {
  public:

	typedef DenseTag tag;
	typedef int Index;
	typedef Scalar ValueType;
	typedef boost::tuples::tuple<Index, Scalar> XTuple;

	const static std::string name;

//	static const bool isIndicator = false;
	enum  { isIndicator = false };
	enum  { isSparse = false };

	inline DenseIterator(const int& start, const int& end)
		: mId(start), mEnd(end) {

	}

	inline DenseIterator(const CompressedDataMatrix<Scalar>& mat, Index column)
	  : mValues(mat.getDataVector(column)),
	    mId(0),
	    //mEnd(mat.getNumberOfRows())
	    mEnd(mat.getDataVectorSTL(column).size())
	    {
		// Do nothing
	}

	inline DenseIterator(const std::vector<int>& vec, Index end)
	: mId(0), mEnd(end) {
		// Do nothing
	}

	inline DenseIterator(const std::vector<int>* vec, Index end)
	    : mId(0), mEnd(end) {
	    // Do nothing
	}

    inline DenseIterator& operator++() { ++mId; return *this; }
	inline DenseIterator operator++(int) { auto rtn = *this; ++(*this); return rtn; }

    inline const Scalar value() const { return mValues[mId]; }
    inline Scalar& valueRef() { return const_cast<Scalar&>(mValues[mId]); }

    inline Index index() const { return mId; }
	inline Index nextIndex() const { return mId + 1; }
    inline operator bool() const { return (mId < mEnd); }
	inline bool inRange(const Scalar i) const { return (mId < i); }
    inline Index size() const { return mEnd; }
	inline Scalar multiply(const Scalar x) const { return x * mValues[mId]; }
	inline Scalar multiply(const Scalar x, const Index index) const { return x * mValues[index]; }

	inline const Scalar* getValues() const { return mValues; }

	ReverseDenseIterator<Scalar> reverse() const {
	    return ReverseDenseIterator<Scalar>(*this);
	}

  protected:
    const Scalar* mValues;
    Index mId;
    const Index mEnd;
};

template <typename Scalar>
class ReverseDenseIterator {
public:

    typedef int Index;

    inline ReverseDenseIterator(const DenseIterator<Scalar>& forward)
        : mValues(forward.getValues()), mEnd(forward.size()), mId(mEnd - 1) {}

    inline ReverseDenseIterator& operator--() { --mId; return *this; }
    inline operator bool() const { return (mId >= 0); }

    inline const Scalar value() const { return mValues[mId]; }
    inline Index index() const { return mId; }

    inline Index size() const { return mEnd; }

protected:
    const Scalar* mValues;
    const Index mEnd;
    Index mId;
};

template <typename Scalar>
class ReverseInterceptIterator;

// Iterator for an intercept column
template <typename Scalar>
class InterceptIterator {
  public:

	typedef InterceptTag tag; // TODO Fix!!!
	typedef int Index;
	typedef Scalar ValueType;
	typedef boost::tuples::tuple<Index> XTuple;

	const static std::string name;

	enum  { isIndicator = true };
	enum  { isSparse = false };

	inline InterceptIterator(const int& start, const int& end)
		: mId(start), mEnd(end) {
		// Do nothing
	}

	inline InterceptIterator(const CompressedDataMatrix<Scalar>& mat, Index column)
	  : mId(0), mEnd(mat.getNumberOfRows()){
		// Do nothing
	}

	inline InterceptIterator(const std::vector<int>& vec, Index end)
	: mId(0), mEnd(end) {
		// Do nothing
	}

	inline InterceptIterator(const std::vector<int>* vec, Index end)
	    : mId(0), mEnd(end) {
	    // Do nothing
	}

    inline InterceptIterator& operator++() { ++mId; return *this; }
	inline InterceptIterator operator++(int) { auto rtn = *this; ++(*this); return rtn; }

    inline const int value() const { return 1; }
    inline int valueRef() { return 1; }
    // TODO Confirm optimization of (real) * value() => (real)

    inline Index index() const { return mId; }
	inline Index nextIndex() const { return mId + 1; }
    inline operator bool() const { return (mId < mEnd); }
	inline bool inRange(const Scalar i) const { return (mId < i); }
    inline Index size() const { return mEnd; }
	inline Scalar multiply(const Scalar x) const { return x; }
	inline Scalar multiply(const Scalar x, const Index index) const { return x; }

	ReverseInterceptIterator<Scalar> reverse() const {
	    return ReverseInterceptIterator<Scalar>(*this);
	}

  protected:
    Index mId;
    const Index mEnd;
};

template <typename Scalar>
class ReverseInterceptIterator {
public:

    typedef int Index;

    inline ReverseInterceptIterator(const InterceptIterator<Scalar>& forward)
        : mEnd(forward.size()), mId(mEnd - 1) {}

    inline ReverseInterceptIterator& operator--() { --mId; return *this; }
    inline operator bool() const { return (mId >= 0); }

    inline const Scalar value() const { return static_cast<Scalar>(1); }
    inline Index index() const { return mId; }

    inline Index size() const { return mEnd; }

protected:
    const Index mEnd;
    Index mId;
};

// Iterator for a dense view of an arbitrary column
template <typename Scalar>
class DenseViewIterator {
  public:

	typedef int Index;
    typedef Scalar ValueType;

	enum  { isIndicator = false };
	enum  { isSparse = false };

	inline DenseViewIterator(const CompressedDataMatrix<Scalar>& mat, Index column)
	  : mValues(mat.getDataVector(column)),
	    mId(0), mEnd(mat.getNumberOfRows()){
		// Do nothing
	}

    inline DenseViewIterator& operator++() { ++mId; return *this; }

    inline const Scalar value() const { return mValues[mId]; }
    inline Scalar& valueRef() { return const_cast<Scalar&>(mValues[mId]); }

    inline Index index() const { return mId; }
	inline Index nextIndex() const { return mId + 1; }
    inline operator bool() const { return (mId < mEnd); }
    inline Index size() const { return mEnd; }
	inline Scalar multiply(const Scalar x) const { return x * mValues[mId]; }
	inline Scalar multiply(const Scalar x, const Index index) const { return x * mValues[index]; }


  protected:
    const Scalar* mValues;
    Index mId;
    const Index mEnd;
};

// Generic iterator for a run-time format determined column
template <typename Scalar>
class GenericIterator {
  public:

	typedef int Index;
    typedef Scalar ValueType;

	inline GenericIterator(const CompressedDataMatrix<Scalar>& mat, Index column)
	  : mFormatType(mat.getFormatType(column)),
	    mId(0) {
		if (mFormatType == DENSE) {
			mValues = mat.getDataVector(column);
			mIndices = NULL;
			mEnd = mat.getNumberOfRows();
		} else if (mFormatType == INTERCEPT) {
		    mValues = NULL;
		    mIndices = NULL;
		    mEnd = mat.getNumberOfRows();
		} else {
			if (mFormatType == SPARSE) {
				mValues = mat.getDataVector(column);
			} else {
				mValues = NULL;
			}
			mIndices = mat.getCompressedColumnVector(column);
			mEnd = mat.getNumberOfEntries(column);
		}
	}

    inline GenericIterator& operator++() { ++mId; return *this; }

    inline const Scalar value() const {
    	if (mFormatType == INDICATOR || mFormatType == INTERCEPT) {
    		return static_cast<Scalar>(1);
    	} else {
    		return mValues[mId];
    	}
    }
//    inline Scalar& valueRef() { return const_cast<Scalar&>(mValues[mId]); }

    inline Index index() const {
    	if (mFormatType == DENSE || mFormatType == INTERCEPT) {
    		return mId;
    	} else {
    		return mIndices[mId];
    	}
    }
	inline Index nextIndex() const {
	    if (mFormatType == DENSE || mFormatType == INTERCEPT) {
	        return mId + 1;
	    } else {
	        return mIndices[mId + 1];
	    }
	}
    inline operator bool() const { return (mId < mEnd); }
    inline Index size() const { return mEnd; }
	inline Scalar multiply(const Scalar x) const { return x * value(); }

  protected:
    const FormatType mFormatType;
    Scalar* mValues;
    Index* mIndices;
    Index mId;
    Index mEnd;
};

// Iterator for grouping by another IndicatorIterator
template <class IteratorType, typename Scalar>
class GroupByIterator {
public:

    typedef int Index;
    typedef Scalar ValueType;

    inline GroupByIterator(IteratorType& itMain, IndicatorIterator<Scalar>& _groupBy)
        : iterator(itMain), groupBy(_groupBy) {
        // Find start
        advance();
    }

    inline GroupByIterator& operator++() {
        ++iterator;
        advance();
        return *this;
    }

    inline Scalar value() const {
        return iterator.value();
    }

    inline Index index() const {
        return iterator.index();
    }

    inline Index group() const {
        return static_cast<Index>(groupBy && iterator.index() == groupBy.index());
    }

    inline operator bool() const {
        return iterator;
    }

private:
	inline void advance() {
	    while(groupBy && iterator && groupBy.index() < iterator.index()) {
	        ++groupBy;
	    }
		// while (valid() && iteratorOne.index() != iteratorTwo.index()) {
//     		if (iteratorOne.index() < iteratorTwo.index()) {
//     			++iteratorOne;
//     		} else {
//     			++iteratorTwo;
//     		}
// 		}
	}

    IteratorType& iterator;
    IndicatorIterator<Scalar>& groupBy;
};

// Iterator for the component-wise product of two Iterator
template <class IteratorOneType, class IteratorTwoType, typename Scalar>
class PairProductIterator {
  public:

	typedef int Index;
    typedef Scalar ValueType;

//	enum  { isIndicator = true }; // Need to determine from IteratorTypeOne/IteratorTypeTwo
//	enum  { isSparse = true };

//	inline PairProductIterator(const CompressedDataMatrix& mat, Index column)
//	  : mIndices(mat.getCompressedColumnVector(column)),
//	    mId(0), mEnd(mat.getNumberOfEntries(column)){
//		// Do nothing
//	}
	inline PairProductIterator(IteratorOneType& itOne, IteratorTwoType& itTwo)
		: iteratorOne(itOne), iteratorTwo(itTwo) {
		// Find start
		advance();
	}

    inline PairProductIterator& operator++() {
    	if (valid()) {
    		++iteratorOne;
    		++iteratorTwo;
    		advance();
    	}
    	return *this;
    }

    inline const Scalar value() const {
    	return iteratorOne.value() * iteratorTwo.value();
    }

    inline Index index() const { return iteratorOne.index(); }

    inline bool valid() const {
    	return iteratorOne && iteratorTwo;
    }

//    inline operator bool() const {
//    	return valid();
//    }

  protected:

	inline void advance() {
//		bool valid = this;
		while (valid() && iteratorOne.index() != iteratorTwo.index()) {
    		if (iteratorOne.index() < iteratorTwo.index()) {
    			++iteratorOne;
    		} else {
    			++iteratorTwo;
    		}
		}
	}

    IteratorOneType& iteratorOne;
    IteratorTwoType& iteratorTwo;
};

// template <typename RealType>
// inline RealType multiply()

// const std::string DenseIterator::name = "Den";
// const std::string IndicatorIterator::name = "Ind";
// const std::string SparseIterator::name = "Spa";
// const std::string InterceptIterator::name = "Icp";

} // namespace

#endif // ITERATORS_H
