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



// Iterator for a sparse of indicators column
class IndicatorIterator {
  public:

	typedef IndicatorTag tag;
	typedef real Scalar;
	typedef int Index;
	typedef boost::tuples::tuple<Index> XTuple;

	const static std::string name;

	static const bool isIndicatorStatic = true;
	enum  { isIndicator = true };
	enum  { isSparse = true };

// 	inline IndicatorIterator(const CompressedDataColumn& col)
// 	  : mIndices(col.getColumns()),
// 	    mId(0), mEnd(

	inline IndicatorIterator(const CompressedDataMatrix& mat, Index column)
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
    inline const Scalar value() const { return static_cast<Scalar>(1); }

    inline Index index() const { return mIndices[mId]; }
    inline operator bool() const { return (mId < mEnd); }

  protected:
    const Index* mIndices;
    Index mId;
    const Index mEnd;
};

// Iterator for a sparse column
class SparseIterator {
  public:

	typedef SparseTag tag;
	typedef real Scalar;
	typedef int Index;
	typedef boost::tuples::tuple<Index, Scalar> XTuple;

	const static std::string name;

	static const bool isIndicatorStatic = false;
	enum  { isIndicator = false };
	enum  { isSparse = true };

	inline SparseIterator(const CompressedDataMatrix& mat, Index column)
	  : mValues(mat.getDataVector(column)), mIndices(mat.getCompressedColumnVector(column)),
	    mId(0), mEnd(mat.getNumberOfEntries(column)){
		// Do nothing
	}

	inline SparseIterator(const CompressedDataColumn& column)
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

    inline const Scalar value() const {
//    	cerr << "Oh yes!" << endl;
//    	exit(-1);
    	return mValues[mId]; }
    inline Scalar& valueRef() { return const_cast<Scalar&>(mValues[mId]); }

    inline Index index() const { return mIndices[mId]; }
    inline operator bool() const { return (mId < mEnd); }

  protected:
    const Scalar* mValues;
    const Index* mIndices;
    Index mId;
    const Index mEnd;
};

template <typename IteratorType>
class DenseView {
public:
	typedef real Scalar;
	typedef int Index;

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
    inline operator bool() const { return (mId < mEnd); }

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
	inline operator bool() const { return (mId < mEnd); }

protected:
	Index mId;
	const Index mEnd;
};

// Iterator for a dense column
class DenseIterator {
  public:

	typedef DenseTag tag;
	typedef real Scalar;
	typedef int Index;
	typedef boost::tuples::tuple<Index, Scalar> XTuple;

	const static std::string name;

//	static const bool isIndicator = false;
	enum  { isIndicator = false };
	enum  { isSparse = false };

	inline DenseIterator(const int& start, const int& end)
		: mId(start), mEnd(end) {

	}

	inline DenseIterator(const CompressedDataMatrix& mat, Index column)
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

    inline const Scalar value() const { return mValues[mId]; }
    inline Scalar& valueRef() { return const_cast<Scalar&>(mValues[mId]); }

    inline Index index() const { return mId; }
    inline operator bool() const { return (mId < mEnd); }

  protected:
    const Scalar* mValues;
    Index mId;
    const Index mEnd;
};

// Iterator for an intercept column
class InterceptIterator {
  public:

	typedef InterceptTag tag; // TODO Fix!!!
	typedef real Scalar;
	typedef int Index;
	typedef boost::tuples::tuple<Index> XTuple;

	const static std::string name;

	enum  { isIndicator = true };
	enum  { isSparse = false };

	inline InterceptIterator(const int& start, const int& end)
		: mId(start), mEnd(end) {
		// Do nothing
	}

	inline InterceptIterator(const CompressedDataMatrix& mat, Index column)
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

    inline const int value() const { return 1; }
    inline int valueRef() { return 1; }
    // TODO Confirm optimization of (real) * value() => (real)

    inline Index index() const { return mId; }
    inline operator bool() const { return (mId < mEnd); }

  protected:
    Index mId;
    const Index mEnd;
};

// Iterator for a dense view of an arbitrary column
class DenseViewIterator {
  public:

	typedef real Scalar;
	typedef int Index;

	enum  { isIndicator = false };
	enum  { isSparse = false };

	inline DenseViewIterator(const CompressedDataMatrix& mat, Index column)
	  : mValues(mat.getDataVector(column)),
	    mId(0), mEnd(mat.getNumberOfRows()){
		// Do nothing
	}

    inline DenseViewIterator& operator++() { ++mId; return *this; }

    inline const Scalar value() const { return mValues[mId]; }
    inline Scalar& valueRef() { return const_cast<Scalar&>(mValues[mId]); }

    inline Index index() const { return mId; }
    inline operator bool() const { return (mId < mEnd); }

  protected:
    const Scalar* mValues;
    Index mId;
    const Index mEnd;
};

// Generic iterator for a run-time format determined column
class GenericIterator {
  public:

	typedef real Scalar;
	typedef int Index;

	inline GenericIterator(const CompressedDataMatrix& mat, Index column)
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
    inline operator bool() const { return (mId < mEnd); }

  protected:
    const FormatType mFormatType;
    Scalar* mValues;
    Index* mIndices;
    Index mId;
    Index mEnd;
};

// Iterator for grouping by another IndicatorIterator
template <class IteratorType>
class GroupByIterator {
public:

    typedef real Scalar;
    typedef int Index;

    inline GroupByIterator(IteratorType& itMain, IndicatorIterator& _groupBy)
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
    IndicatorIterator& groupBy;
};

// Iterator for the component-wise product of two Iterator
template <class IteratorOneType, class IteratorTwoType>
class PairProductIterator {
  public:

	typedef real Scalar;
	typedef int Index;

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

// const std::string DenseIterator::name = "Den";
// const std::string IndicatorIterator::name = "Ind";
// const std::string SparseIterator::name = "Spa";
// const std::string InterceptIterator::name = "Icp";


} // namespace

#endif // ITERATORS_H
