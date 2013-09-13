#ifndef ITERATORS_H
#define ITERATORS_H

#include "CompressedDataMatrix.h"

namespace bsccs {

/**
 * Iterators for dense, sparse and indicator vectors.  Each can be passed as a
 * template parameter to functions for compile-time specialization and optimization.
 * If performance is not an issue, GenericIterator wraps all three formats into a
 * single run-time determined iterator
 */

// Iterator for a sparse of indicators column
class IndicatorIterator {
  public:

	typedef real Scalar;
	typedef int Index;

//	static const bool isIndicator = true;
	enum  { isIndicator = true };
	enum  { isSparse = true };

	inline IndicatorIterator(const CompressedDataMatrix& mat, Index column)
	  : mIndices(mat.getCompressedColumnVector(column)),
	    mId(0), mEnd(mat.getNumberOfEntries(column)){
		// Do nothing
	}

	inline IndicatorIterator(const std::vector<int>& vec, Index max = 0)
	: mIndices(vec.data()), mId(0), mEnd(vec.size()) {
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

	typedef real Scalar;
	typedef int Index;

//	static const bool isIndicator = false;
	enum  { isIndicator = false };
	enum  { isSparse = true };

	inline SparseIterator(const CompressedDataMatrix& mat, Index column)
	  : mValues(mat.getDataVector(column)), mIndices(mat.getCompressedColumnVector(column)),
	    mId(0), mEnd(mat.getNumberOfEntries(column)){
		// Do nothing
	}

	inline SparseIterator(const std::vector<int>& vec, Index max = 0)
	: mIndices(vec.data()), mId(0), mEnd(vec.size()) {
		// Do nothing
	}

    inline SparseIterator& operator++() { ++mId; return *this; }

    inline const Scalar& value() const {
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

	typedef real Scalar;
	typedef int Index;

//	static const bool isIndicator = false;
	enum  { isIndicator = false };
	enum  { isSparse = false };
	
	inline DenseIterator(const int& start, const int& end)
		: mId(start), mEnd(end) {
		
	}

	inline DenseIterator(const CompressedDataMatrix& mat, Index column)
	  : mValues(mat.getDataVector(column)),
	    mId(0), mEnd(mat.getNumberOfRows()){
		// Do nothing
	}

	inline DenseIterator(const std::vector<int>& vec, Index end)
	: mId(0), mEnd(end) {
		// Do nothing
	}

    inline DenseIterator& operator++() { ++mId; return *this; }

    inline const Scalar& value() const { return mValues[mId]; }
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

	typedef real Scalar;
	typedef int Index;

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

    inline const Scalar& value() const { return mValues[mId]; }
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
    	if (mFormatType == INDICATOR) {
    		return static_cast<Scalar>(1);
    	} else {
    		return mValues[mId];
    	}
    }
//    inline Scalar& valueRef() { return const_cast<Scalar&>(mValues[mId]); }

    inline Index index() const {
    	if (mFormatType == DENSE) {
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

} // namespace

#endif // ITERATORS_H
