#ifndef ITERATORS_TUPLEIT_HH
#define ITERATORS_TUPLEIT_HH
 
#include <iterator>
#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include <new>
#include "boost/tuple/tuple.hpp"
#include "boost/tuple/tuple_comparison.hpp"
#include "boost/utility.hpp"
#include "boost/type_traits.hpp"
#include "boost/optional.hpp" // for aligned_storage
#include <memory>
 
namespace iterators
{
    namespace detail
    {
        void preincrementTuple(boost::tuples::null_type)
        {
        }
 
        template<typename TupleType>
        void preincrementTuple(TupleType& lhs)
        {
            preincrementTuple(lhs.get_tail());
            ++(lhs.template get<0>());
        }
 
        void predecrementTuple(boost::tuples::null_type)
        {
        }
 
        template<typename TupleType>
        void predecrementTuple(TupleType& lhs)
        {
            predecrementTuple(lhs.get_tail());
            --(lhs.template get<0>());
        }
 
        template<typename difference_type>
        void addToTuple(boost::tuples::null_type,difference_type)
        {
        }
 
        template<typename difference_type,typename TupleType>
        void addToTuple(TupleType& lhs,difference_type diff)
        {
            addToTuple(lhs.get_tail(),diff);
            lhs.template get<0>()+=diff;
        }
 
        template<typename difference_type>
        void subFromTuple(boost::tuples::null_type,difference_type)
        {
        }
 
        template<typename difference_type,typename TupleType>
        void subFromTuple(TupleType& lhs,difference_type diff)
        {
            subFromTuple(lhs.get_tail(),diff);
            lhs.template get<0>()-=diff;
        }
 
        template<typename difference_type,typename TupleType>
        difference_type diffTuples(TupleType const& lhs,TupleType const& rhs);
 
        template<typename difference_type,typename TupleType>
        struct DiffTupleHelper
        {
            static difference_type doDiff(TupleType const& lhs,TupleType const& rhs)
            {
                difference_type res1=lhs.template get<0>()-rhs.template get<0>();
                difference_type res2=diffTuples<difference_type>(lhs.get_tail(),rhs.get_tail());
               
                if(res1==res2)
                {
                    return res1;
                }
               
                throw std::logic_error("The iterators in the tuples are mismatched");
            }
        };
 
        template<typename difference_type,typename ValueType>
        struct DiffTupleHelper<difference_type,boost::tuples::cons<ValueType,boost::tuples::null_type> >
        {
            static difference_type doDiff(boost::tuples::cons<ValueType,boost::tuples::null_type> const& lhs,boost::tuples::cons<ValueType,boost::tuples::null_type>  const& rhs)
            {
                return lhs.template get<0>()-rhs.template get<0>();
            }
        };
   
        template<typename difference_type,typename TupleType>
        difference_type diffTuples(TupleType const& lhs,TupleType const& rhs)
        {
            return DiffTupleHelper<difference_type,TupleType>::doDiff(lhs,rhs);
        }
 
   
 
        template<typename SourceTuple>
        struct MakeTupleTypeWithReferences
        {
            typedef MakeTupleTypeWithReferences<typename SourceTuple::tail_type> TailTupleTypeBuilder;
            typedef typename TailTupleTypeBuilder::Type TailTupleType;
            typedef boost::tuples::cons<typename boost::add_reference<typename SourceTuple::head_type>::type,
                                        TailTupleType> Type;
 
            template<typename Tuple>
            static Type makeTuple(Tuple& source)
            {
                return Type(source.get_head(),TailTupleTypeBuilder::makeTuple(source.get_tail()));
            }
        };
   
        template<>
        struct MakeTupleTypeWithReferences<boost::tuples::null_type>
        {
            typedef boost::tuples::null_type Type;
 
            static Type makeTuple(boost::tuples::null_type)
            {
                return Type();
            }
        };
 
        typedef char Tiny;
        struct Small
        {
            Tiny dummy[2];
        };
        struct Medium
        {
            Small dummy[2];
        };
        struct Large
        {
            Medium dummy[2];
        };
        struct Huge
        {
            Large dummy[2];
        };
 
        template<unsigned>
        struct CategoryMap
        {
            typedef void Type;
        };
       
       
 
//     Tiny categoryCheck(std::output_iterator_tag*);
        Small categoryCheck(std::input_iterator_tag*);
        Medium categoryCheck(std::forward_iterator_tag*);
        Large categoryCheck(std::bidirectional_iterator_tag*);
        Huge categoryCheck(std::random_access_iterator_tag*);
 
 
//     template<>
//     struct CategoryMap<sizeof(Tiny)>
//     {
//      typedef std::output_iterator_tag Type;
//     };
 
        template<>
        struct CategoryMap<sizeof(Small)>
        {
            typedef std::input_iterator_tag Type;
        };
   
        template<>
        struct CategoryMap<sizeof(Medium)>
        {
            typedef std::forward_iterator_tag Type;
        };
 
        template<>
        struct CategoryMap<sizeof(Large)>
        {
            typedef std::bidirectional_iterator_tag Type;
        };
        template<>
        struct CategoryMap<sizeof(Huge)>
        {
            typedef std::random_access_iterator_tag Type;
        };
 
        template<typename Cat1,typename Cat2>
        struct CommonCategory
        {
        private:
            enum
            {categorySize=sizeof(::iterators::detail::categoryCheck(false?(Cat1*)0:(Cat2*)0))
            };
        public:
            typedef typename CategoryMap<categorySize>::Type Type;
        };
       
        // specializations
        template<typename Cat>
        struct CommonCategory<std::output_iterator_tag,Cat>
        {
            typedef std::output_iterator_tag Type;
        };
        template<typename Cat>
        struct CommonCategory<Cat,std::output_iterator_tag>
        {
            typedef std::output_iterator_tag Type;
        };
        template<>
        struct CommonCategory<std::output_iterator_tag,std::output_iterator_tag>
        {
            typedef std::output_iterator_tag Type;
        };
        template<>
        struct CommonCategory<std::input_iterator_tag,std::output_iterator_tag>
        {
            // no Type, because error
        };
        template<>
        struct CommonCategory<std::output_iterator_tag,std::input_iterator_tag>
        {
            // no Type, because error
        };
 
        void derefAndWrite(boost::tuples::null_type,boost::tuples::null_type)
        {}
 
        template<typename IterTuple,typename SourceTuple>
        void derefAndWrite(IterTuple& iters,SourceTuple const& source)
        {
            *iters.get_head()=source.get_head();
            derefAndWrite(iters.get_tail(),source.get_tail());
        }
       
    }
 
    // An OutputTuple holds a tuple of references to iterators, and writes to them on assignment
    template<typename IterTuple>
    struct OutputTuple:
        public detail::MakeTupleTypeWithReferences<IterTuple>::Type,
        boost::noncopyable
    {
    private:
        typedef detail::MakeTupleTypeWithReferences<IterTuple> BaseTypeBuilder;
        typedef typename BaseTypeBuilder::Type BaseType;
    public:
        OutputTuple(IterTuple& iters):
            BaseType(BaseTypeBuilder::makeTuple(iters))
        {}
 
        template<typename SomeTuple>
        OutputTuple& operator=(const SomeTuple& other)
        {
            detail::derefAndWrite(static_cast<BaseType&>(*this),other);
            return *this;
        }
    };
 
    // An OwningRefTuple holds a tuple of references,
    // which may point to data within the tuple, or external to it
 
    namespace detail
    {
        struct PreserveReferences
        {};
 
        template<typename OwnedType>
        struct OwningBase
        {
            std::auto_ptr<OwnedType> tupleBuf;
 
            OwningBase()
            {}
           
            template<typename SomeType>
            OwningBase(SomeType &source):
                tupleBuf(new OwnedType(source))
            {}
           
        };
    }
 
    template<typename TupleType>
    struct OwningRefTuple:
        private detail::OwningBase<TupleType>,
        public detail::MakeTupleTypeWithReferences<TupleType>::Type
    {
    private:
        typedef detail::MakeTupleTypeWithReferences<TupleType> BaseTypeBuilder;
        typedef typename BaseTypeBuilder::Type BaseType;
        typedef detail::OwningBase<TupleType> OwningBaseType;
    public:
 
        typedef typename BaseType::head_type head_type;
        typedef typename BaseType::tail_type tail_type;
 
    private:
        typedef TupleType OwnedTuple;
 
        OwnedTuple* getTuplePtr()
        {
            return this->tupleBuf.get();
        }
    public:
        // copy from other types of tuples too
        template<typename SomeTuple>
        OwningRefTuple(const SomeTuple& other):
            OwningBaseType(other),BaseType(BaseTypeBuilder::makeTuple(*getTuplePtr()))
        {
        }
        // copying copies values by default
        OwningRefTuple(const OwningRefTuple& other):
            OwningBaseType(other),BaseType(BaseTypeBuilder::makeTuple(*getTuplePtr()))
        {
        }
 
        // allow user to specify
        // whether to preserve references
        template<typename SomeTuple>
        OwningRefTuple(SomeTuple& other,detail::PreserveReferences const&):
            BaseType(BaseTypeBuilder::makeTuple(other))
        {
        }
 
        // assignment assigns to referenced values
        template<typename SomeTuple>
        OwningRefTuple& operator=(const SomeTuple& other)
        {
            BaseType::operator=(other);
            return *this;
        }
        OwningRefTuple& operator=(const OwningRefTuple& other)
        {
            BaseType::operator=(other);
            return *this;
        }
    };
 
    namespace detail
    {
        template<typename IterTuple>
        struct DerefIterTupleHelperKeepRef
        {
            typedef boost::tuples::cons<typename boost::add_reference<typename std::iterator_traits<typename IterTuple::head_type>::value_type>::type,
                                        typename DerefIterTupleHelperKeepRef<typename IterTuple::tail_type>::Type> Type;
        };
 
        template<>
        struct DerefIterTupleHelperKeepRef<boost::tuples::null_type>
        {
            typedef boost::tuples::null_type Type;
        };
 
        template<>
        struct DerefIterTupleHelperKeepRef<const boost::tuples::null_type>
        {
            typedef boost::tuples::null_type Type;
        };
 
 
        template<typename IterTuple>
        struct DerefIterTupleHelperNoRef
        {
            typedef boost::tuples::cons<typename std::iterator_traits<typename IterTuple::head_type>::value_type,
                                        typename DerefIterTupleHelperNoRef<typename IterTuple::tail_type>::Type> Type;
        };
 
        template<>
        struct DerefIterTupleHelperNoRef<boost::tuples::null_type>
        {
            typedef boost::tuples::null_type Type;
        };
 
        template<>
        struct DerefIterTupleHelperNoRef<const boost::tuples::null_type>
        {
            typedef boost::tuples::null_type Type;
        };
 
        boost::tuples::null_type derefIterTupleKeepRef(boost::tuples::null_type const& iters)
        {
            return iters;
        }
        template<typename IterTuple>
        const typename DerefIterTupleHelperKeepRef<IterTuple>::Type derefIterTupleKeepRef(IterTuple& iters)
        {
            return typename DerefIterTupleHelperKeepRef<IterTuple>::Type(*iters.template get<0>(),derefIterTupleKeepRef(iters.get_tail()));
        }
 
        boost::tuples::null_type derefIterTupleNoRef(boost::tuples::null_type const& iters)
        {
            return iters;
        }
        template<typename IterTuple>
        typename DerefIterTupleHelperNoRef<IterTuple>::Type derefIterTupleNoRef(IterTuple& iters)
        {
            return typename DerefIterTupleHelperNoRef<IterTuple>::Type(*iters.template get<0>(),derefIterTupleNoRef(iters.get_tail()));
        }
 
        // Define, construct and destroy the appropriate value_type for
        // the given iterator category
        template<typename Category,typename IterTuple>
        struct ValueForCategory
        {
        private:
            typedef typename IterTuple::head_type HeadIterType;
            typedef typename IterTuple::tail_type TailTupleType;
            typedef typename std::iterator_traits<HeadIterType>::value_type HeadValueType;
            typedef typename ValueForCategory<Category,TailTupleType>::ValueTuple TailValueTuple;
 
        public:
            typedef boost::tuples::cons<HeadValueType,TailValueTuple> ValueTuple;
           
            typedef OwningRefTuple<ValueTuple> value_type;
            typedef value_type Type;
           
            static void construct(Type* p,IterTuple const& iters)
            {
                // don't copy values, keep as references
                new (p) Type(derefIterTupleKeepRef(iters),::iterators::detail::PreserveReferences());
            }
           
            static void destruct(Type* p)
            {
                p->~OwningRefTuple<ValueTuple>();
            }
        };
 
        template<typename Category>
        struct ValueForCategory<Category,boost::tuples::null_type>
        {
        private:
        public:
            typedef boost::tuples::null_type ValueTuple;
        };
 
        template<typename IterTuple>
        struct ValueForCategory<std::input_iterator_tag,IterTuple>
        {
        private:
            typedef typename IterTuple::head_type HeadIterType;
            typedef typename IterTuple::tail_type TailTupleType;
            typedef typename std::iterator_traits<HeadIterType>::value_type HeadValueType;
            typedef typename ValueForCategory<std::input_iterator_tag,TailTupleType>::ValueTuple TailValueTuple;
 
        public:
            typedef boost::tuples::cons<HeadValueType,TailValueTuple> ValueTuple;
           
            typedef OwningRefTuple<ValueTuple> value_type;
            typedef value_type Type;
           
            static void construct(Type* p,IterTuple const& iters)
            {
                // copy values
                new (p) Type(derefIterTupleNoRef(iters));
            }
           
            static void destruct(Type* p)
            {
                p->~OwningRefTuple<ValueTuple>();
            }
        };
 
        template<>
        struct ValueForCategory<std::input_iterator_tag,boost::tuples::null_type>
        {
        private:
        public:
            typedef boost::tuples::null_type ValueTuple;
        };
 
        template<typename IterTuple>
        struct ValueForCategory<std::output_iterator_tag,IterTuple>
        {
        public:
            typedef OutputTuple<IterTuple> value_type;
            typedef value_type Type;
           
            static void construct(Type* p,IterTuple& iters)
            {
                // copy values
                new (p) Type(iters);
            }
           
            static void destruct(Type* p)
            {
                p->~OutputTuple<IterTuple>();
            }
        };
 
        template<>
        struct ValueForCategory<std::output_iterator_tag,boost::tuples::null_type>
        {
        private:
        public:
        };
 
 
        template<typename Category,typename IterTuple>
        struct VFCSelector
        {
            typedef ValueForCategory<Category,IterTuple> Type;
        };
 
        // Select the iterator_category and value_type for our TupleIt
        template<typename IterTuple>
        struct TupleItHelper
        {
            typedef typename IterTuple::head_type HeadIterType;
            typedef typename IterTuple::tail_type TailTupleType;
           
            typedef typename std::iterator_traits<HeadIterType>::iterator_category Cat1;
            typedef typename TupleItHelper<TailTupleType>::iterator_category Cat2;
 
            typedef typename CommonCategory<Cat1,Cat2>::Type iterator_category;
            typedef typename VFCSelector<iterator_category,IterTuple>::Type ValueTypeDef;
            typedef typename ValueTypeDef::value_type value_type;
            typedef typename ValueTypeDef::Type DeRefType;
 
            typedef DeRefType& reference;
            typedef DeRefType* pointer;
 
            typedef std::ptrdiff_t difference_type;
 
            typedef std::iterator<iterator_category,value_type,difference_type,pointer,reference> IteratorType;
 
            static void construct(DeRefType* p,IterTuple& iters)
            {
                ValueTypeDef::construct(p,iters);
            }
           
            static void destruct(DeRefType* p)
            {
                ValueTypeDef::destruct(p);
            }
        };
 
        template<>
        struct TupleItHelper<boost::tuples::null_type>
        {
            typedef std::random_access_iterator_tag iterator_category;
        };
    }
 
    // the actual Tuple Iterator itself
    template<typename IterTuple>
    struct TupleIt:
        public detail::TupleItHelper<IterTuple>::IteratorType
    {
    private:
        typedef detail::TupleItHelper<IterTuple> TupleDefs;
    public:
        typedef typename TupleDefs::iterator_category iterator_category;
        typedef typename TupleDefs::value_type value_type;
        typedef typename TupleDefs::difference_type difference_type;
        typedef typename TupleDefs::reference reference;
        typedef typename TupleDefs::pointer pointer;
    private:
        pointer getValuePtr() const
        {
            return reinterpret_cast<pointer>(dataCache.address());
        }
   
        void emptyCache() const
        {
            if(cacheInitialized)
            {
                TupleDefs::destruct(getValuePtr());
                cacheInitialized=false;
            }
        }
 
        void initCache() const
        {
            emptyCache();
            TupleDefs::construct(getValuePtr(),iters);
            cacheInitialized=true;
        }
   
   
    public:
   
        TupleIt(IterTuple iters_):
            iters(iters_),cacheInitialized(false)
        {}
        template<typename OtherIterTuple>
        TupleIt(const TupleIt<OtherIterTuple>& other):
            iters(other.iters),cacheInitialized(false)
        {}
        TupleIt(const TupleIt& other):
            iters(other.iters),cacheInitialized(false)
        {}
        TupleIt():
            iters(),cacheInitialized(false)
        {}
   
 
        ~TupleIt()
        {
            emptyCache();
        }
 
        void swap(TupleIt& other)
        {
            using std::swap;
       
            swap(iters,other.iters);
        }
 
        TupleIt& operator=(TupleIt const& other)
        {
            emptyCache();
            iters=other.iters;
            return *this;
        }
 
        // Input Iterator requirements
        reference operator*() const
        {
            initCache();
            return *getValuePtr();
        }
 
        pointer operator->() const
        {
            initCache();
            return getValuePtr();
        }
 
        friend bool operator==(const TupleIt& lhs,const TupleIt& rhs)
        {
            return lhs.iters==rhs.iters;
        }
 
        friend bool operator!=(const TupleIt& lhs,const TupleIt& rhs)
        {
            return lhs.iters!=rhs.iters;
        }
 
        // Forward Iterator requirements
        TupleIt& operator++()
        {
            detail::preincrementTuple(iters);
            return *this;
        }
   
        TupleIt operator++(int)
        {
            TupleIt temp(*this);
            ++*this;
            return temp;
        }
 
        // Bidirectional Iterator requirements
        TupleIt& operator--()
        {
            detail::predecrementTuple(iters);
            return *this;
        }
   
        TupleIt operator--(int)
        {
            TupleIt temp(*this);
            --*this;
            return temp;
        }
   
        // Random-Access Iterator requirements
        TupleIt& operator+=(difference_type n)
        {
            detail::addToTuple(iters,n);
            return *this;
        }
 
        TupleIt& operator-=(difference_type n)
        {
            detail::subFromTuple(iters,n);
            return *this;
        }
 
        friend difference_type operator-(const TupleIt& a,const TupleIt& b)
        {
            return detail::diffTuples<difference_type>(a.iters,b.iters);
        }
 
        value_type operator[](difference_type n) const
        {
            return *(*this+n);
        }
 
    private:
        // everything is mutable so we can modify it without affecting const correctness
        // of client code
        mutable IterTuple iters;
        mutable boost::optional_detail::aligned_storage<typename TupleDefs::DeRefType> dataCache;
        mutable bool cacheInitialized;
    };
 
    // more random-access iterator requirements
    template<typename IterTuple>
    TupleIt<IterTuple> operator+(std::ptrdiff_t n,TupleIt<IterTuple> temp)
    {
        temp+=n;
        return temp;
    }
 
    template<typename IterTuple>
    TupleIt<IterTuple> operator+(TupleIt<IterTuple> temp,std::ptrdiff_t n)
    {
        temp+=n;
        return temp;
    }
 
    template<typename IterTuple>
    TupleIt<IterTuple> operator-(TupleIt<IterTuple> temp,std::ptrdiff_t n)
    {
        temp-=n;
        return temp;
    }
 
    template<typename IterTuple,typename IterTuple2>
    bool operator<(const TupleIt<IterTuple>& a,const TupleIt<IterTuple2>& b)
    {
        return (b-a)>0;
    }
 
    template<typename IterTuple,typename IterTuple2>
    bool operator>(const TupleIt<IterTuple>& a,const TupleIt<IterTuple2>& b)
    {
        return b<a;
    }
 
    template<typename IterTuple,typename IterTuple2>
    bool operator>=(const TupleIt<IterTuple>& a,const TupleIt<IterTuple2>& b)
    {
        return !(b<a);
    }
 
    template<typename IterTuple,typename IterTuple2>
    bool operator<=(const TupleIt<IterTuple>& a,const TupleIt<IterTuple2>& b)
    {
        return !(b>a);
    }
 
    // implementation of swap and iter_swap
    template<typename IterTuple>
    void swap(TupleIt<IterTuple>& lhs,TupleIt<IterTuple>& rhs)
    {
        lhs.swap(rhs);
    }
   
//     template<typename IterTuple,IterTuple2>
//     void iter_swap(const TupleIt<IterTuple>& lhs,const TupleIt<IterTuple2>& rhs)
//     {
//      lhs.iter_swap(rhs);
//     }
 
    template<typename Iter1,typename Iter2>
    TupleIt<typename boost::tuples::tuple<Iter1,Iter2> > makeTupleIterator(Iter1 i1,Iter2 i2)
    {
        return TupleIt<typename boost::tuples::tuple<Iter1,Iter2> >(boost::make_tuple(i1,i2));
    }
 
    template<typename Iter1,typename Iter2,typename Iter3>
    TupleIt<typename boost::tuples::tuple<Iter1,Iter2,Iter3> > makeTupleIterator(Iter1 i1,Iter2 i2,Iter3 i3)
    {
        return TupleIt<typename boost::tuples::tuple<Iter1,Iter2,Iter3> >(boost::make_tuple(i1,i2,i3));
    }
 
    template<typename Iter1,typename Iter2,typename Iter3,typename Iter4>
    TupleIt<typename boost::tuples::tuple<Iter1,Iter2,Iter3,Iter4> > makeTupleIterator(Iter1 i1,Iter2 i2,Iter3 i3,Iter4 i4)
    {
        return TupleIt<typename boost::tuples::tuple<Iter1,Iter2,Iter3,Iter4> >(boost::make_tuple(i1,i2,i3,i4));
    }
   
}
 
#endif