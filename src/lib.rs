use std::fmt;
use std::marker; //::PhantomData;
use std::ops;

/// Represents a single axis dimension of a multi dimensional shape
pub trait Dim: 'static + Sized + Clone + Copy + fmt::Debug + PartialEq + Eq + Into<usize> {}

/// Represents a dimension whose length is known at compile time
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub struct Const<const N: usize>;

impl<const N: usize> fmt::Debug for Const<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_const", N)
    }
}

impl<const M: usize> Dim for Const<M> {}
impl<const M: usize> From<Const<M>> for usize {
    fn from(_: Const<M>) -> Self {
        M
    }
}

pub trait Index: 'static + Sized + Clone + Copy + fmt::Debug {
    const NUM_DIMENSIONS: usize;
    type Dyn: IndexDyn;

    // Required methods
    fn d(&self) -> Self::Dyn;

    // Provided methods
    fn num_elements(&self) -> usize {
        self.axis_iter().product()
    }

    fn default_strides(&self) -> Self::Dyn {
        let mut result = Self::Dyn::zero();
        let mut acc = 1;
        for (r, d) in result.iter_mut().rev().zip(self.axis_iter().rev()) {
            *r = acc;
            acc *= d;
        }
        result
    }

    fn shape_mismatch_fail(&self, other: &Self) {
        for (i, s) in self.axis_iter().zip(other.axis_iter()) {
            if i != s {
                panic!("Shapes do not match: expected={self:?} got={other:?}");
            }
        }
    }

    fn out_of_bounds_fail(&self, idx: &Self::Dyn) {
        for (i, s) in idx.axis_iter().zip(self.axis_iter()) {
            if i >= s {
                panic!("Index out of bounds: index={idx:?} shape={self:?}");
            }
        }
    }

    fn to_i(&self, strides: &Self) -> usize {
        strides
            .axis_iter()
            .zip(self.axis_iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    fn axis_iter(self) -> impl Iterator<Item = usize> + DoubleEndedIterator + ExactSizeIterator {
        IntoIterator::into_iter(self.d())
    }
}

pub trait IndexDyn:
    Index<Dyn = Self>
    + ops::IndexMut<usize>
    + IntoIterator<Item = usize, IntoIter: DoubleEndedIterator + ExactSizeIterator>
{
    fn zero() -> Self;
    fn iter_mut<'a>(&'a mut self) -> std::slice::IterMut<'a, usize>;
}

/*
pub trait Index:
    Clone
    + Copy
    + fmt::Debug
    + IntoIterator<Item = usize, IntoIter: DoubleEndedIterator + ExactSizeIterator>
    + ops::Index<usize, Output = usize>
    + ops::IndexMut<usize>
    + PartialEq<Self>
    + Eq
{
    // Required methods:
    fn zero() -> Self;
    fn iter_mut<'a>(&'a mut self) -> std::slice::IterMut<'a, usize>;

    // Provided methods:
    fn in_bounds<S: Shape<Index = Self>>(&self, shape: &S) -> bool {
        self.into_iter()
            .zip(shape.value().into_iter())
            .all(|(i, s)| i < s)
    }

    fn out_of_bounds_fail<S: Shape<Index = Self>>(&self, shape: &S) {
        for (i, s) in self.into_iter().zip(shape.value().into_iter()) {
            if i >= s {
                panic!("Index out of bounds: index={self:?} shape={shape:?}");
            }
        }
    }

    fn to_i(&self, strides: &Self) -> usize {
        strides
            .into_iter()
            .zip(self.into_iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}
*/

impl<const N: usize> Index for [usize; N] {
    const NUM_DIMENSIONS: usize = N;
    type Dyn = [usize; N];

    fn d(&self) -> Self::Dyn {
        *self
    }
}

impl<const N: usize> IndexDyn for [usize; N] {
    fn zero() -> Self {
        [0; N]
    }

    fn iter_mut<'a>(&'a mut self) -> std::slice::IterMut<'a, usize> {
        <[usize]>::iter_mut(self)
    }
}

/*
struct IndexIter<const N: usize> {
    data: [usize; N],
    cur: usize,
    cur_back: usize,
}

impl<const N: usize> Iterator for IndexIter<N> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.cur + self.cur_back > N {
            return None;
        }
        let val = unsafe { self.data.get_unchecked(self.cur) };
        self.cur += 1;
        Some(val)
    }
}

impl<const N: usize> DoubleEndedIterator for IndexIter<N> {
    // Required method
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.cur + self.cur_back > N {
            return None;
        }
        let val = unsafe { self.data.get_unchecked(N - self.cur_back) };
        self.cur_back += 1;
        Some(val)
    }
}
*/

impl<D1: Dim> Index for (D1,) {
    const NUM_DIMENSIONS: usize = 1;
    type Dyn = [usize; 1];

    fn d(&self) -> Self::Dyn {
        [self.0.into()]
    }
}

impl<D1: Dim, D2: Dim> Index for (D1, D2) {
    const NUM_DIMENSIONS: usize = 2;
    type Dyn = [usize; 2];

    fn d(&self) -> Self::Dyn {
        [self.0.into(), self.1.into()]
    }
}

/////////////////////////////////////////////

pub trait DefiniteRange: Sized {
    type Item: IndexDyn;

    fn first(&self) -> Option<Self::Item>;
    fn next(&self, cur: Self::Item) -> Option<Self::Item>;

    fn nd_iter(self) -> RangeIter<Self> {
        let cur = self.first();
        RangeIter { range: self, cur }
    }
}

impl<I: Index> DefiniteRange for ops::RangeTo<I> {
    type Item = I::Dyn;

    fn first(&self) -> Option<Self::Item> {
        self.end
            .axis_iter()
            .all(|n| n > 0)
            .then_some(I::Dyn::zero())
    }

    fn next(&self, mut cur: Self::Item) -> Option<Self::Item> {
        for (i, n) in cur.iter_mut().rev().zip(self.end.axis_iter().rev()) {
            *i += 1;
            if *i < n {
                return Some(cur);
            } else {
                // Iteration finished in this axis
                *i = 0;
            }
        }
        None
    }
}

impl<I: Index> DefiniteRange for ops::RangeToInclusive<I> {
    type Item = I::Dyn;

    fn first(&self) -> Option<Self::Item> {
        Some(I::Dyn::zero())
    }

    fn next(&self, mut cur: Self::Item) -> Option<Self::Item> {
        for (i, n) in cur.iter_mut().rev().zip(self.end.axis_iter().rev()) {
            *i += 1;
            if *i <= n {
                return Some(cur);
            } else {
                // Iteration finished in this axis
                *i = 0;
            }
        }
        None
    }
}

impl<I: Index> DefiniteRange for ops::Range<I> {
    type Item = I::Dyn;

    fn first(&self) -> Option<Self::Item> {
        self.start
            .axis_iter()
            .zip(self.end.axis_iter())
            .all(|(s, e)| e > s)
            .then_some(self.start.d())
    }

    fn next(&self, mut cur: Self::Item) -> Option<Self::Item> {
        for (i, (s, e)) in cur
            .iter_mut()
            .rev()
            .zip(self.start.axis_iter().rev().zip(self.end.axis_iter().rev()))
        {
            *i += 1;
            if *i < e {
                return Some(cur);
            } else {
                // Iteration finished in this axis
                *i = s;
            }
        }
        None
    }
}

impl<I: Index> DefiniteRange for ops::RangeInclusive<I> {
    type Item = I::Dyn;

    fn first(&self) -> Option<Self::Item> {
        self.start()
            .axis_iter()
            .zip(self.end().axis_iter())
            .all(|(s, e)| e >= s)
            .then_some(self.start().d())
    }

    fn next(&self, mut cur: Self::Item) -> Option<Self::Item> {
        for (i, (s, e)) in cur.iter_mut().rev().zip(
            self.start()
                .axis_iter()
                .rev()
                .zip(self.end().axis_iter().rev()),
        ) {
            *i += 1;
            if *i <= e {
                return Some(cur);
            } else {
                // Iteration finished in this axis
                *i = s;
            }
        }
        None
    }
}

/////////////////////////////////////////////

// Wrappers for array data
// Three kinds: owned, ref, mut

#[derive(Debug, Clone)]
pub struct Array<S: Index, E, D> {
    shape: S,
    element: marker::PhantomData<E>,
    offset: usize,
    strides: S::Dyn,
    data: D,
}

#[derive(Debug)]
pub struct View<'a, S: Index, E, D> {
    shape: S,
    element: marker::PhantomData<E>,
    offset: usize,
    strides: S::Dyn,
    data: &'a D,
}

#[derive(Debug)]
pub struct ViewMut<'a, S: Index, E, D> {
    shape: S,
    element: marker::PhantomData<E>,
    offset: usize,
    strides: S::Dyn,
    data: &'a mut D,
}

// Get a view
pub trait IntoView {
    type Shape: Index;
    type Element;
    type Data;

    fn view(&self) -> View<Self::Shape, Self::Element, Self::Data>;
}

impl<S: Index, E, D> IntoView for Array<S, E, D> {
    type Shape = S;
    type Element = E;
    type Data = D;

    fn view(&self) -> View<Self::Shape, Self::Element, Self::Data> {
        View {
            shape: self.shape.clone(),
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides,
            data: &self.data,
        }
    }
}

impl<S: Index, E, D> IntoView for View<'_, S, E, D> {
    type Shape = S;
    type Element = E;
    type Data = D;

    fn view(&self) -> View<Self::Shape, Self::Element, Self::Data> {
        View {
            shape: self.shape.clone(),
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides,
            data: self.data,
        }
    }
}

impl<S: Index, E, D> IntoView for ViewMut<'_, S, E, D> {
    type Shape = S;
    type Element = E;
    type Data = D;

    fn view(&self) -> View<Self::Shape, Self::Element, Self::Data> {
        View {
            shape: self.shape.clone(),
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides,
            data: self.data,
        }
    }
}

// Get a mutable view
pub trait IntoViewMut: IntoView {
    fn view_mut(&mut self) -> ViewMut<'_, Self::Shape, Self::Element, Self::Data>;
}

impl<S: Index, E, D> IntoViewMut for Array<S, E, D> {
    fn view_mut(&mut self) -> ViewMut<'_, Self::Shape, Self::Element, Self::Data> {
        ViewMut {
            shape: self.shape.clone(),
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides,
            data: &mut self.data,
        }
    }
}

impl<S: Index, E, D> IntoViewMut for ViewMut<'_, S, E, D> {
    fn view_mut(&mut self) -> ViewMut<'_, Self::Shape, Self::Element, Self::Data> {
        ViewMut {
            shape: self.shape.clone(),
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides,
            data: self.data,
        }
    }
}

/////////////////////////////////////////////

macro_rules! impl_view_methods {
    ($struct:ident $(<$lt:lifetime>)?) => {
        impl<S: Index, E, D: AsRef<[E]>> $struct<$($lt,)? S, E, D> {
            pub unsafe fn get_unchecked(&self, idx: S::Dyn) -> &E {
                self.data.as_ref().get_unchecked(self.offset + idx.to_i(&self.strides))
            }
        }

        impl<S: Index, E, D: AsRef<[E]>> ops::Index<S::Dyn> for $struct<$($lt,)? S, E, D> {
            type Output = E;

            fn index(&self, idx: S::Dyn) -> &E {
                self.shape.out_of_bounds_fail(&idx);
                unsafe { self.get_unchecked(idx) }
            }
        }

        impl<'a, S: Index, E, D: AsRef<[E]>> IntoIterator for &'a $struct<$($lt,)? S, E, D> {
            type Item = &'a E;
            type IntoIter = NdIter<'a, S, E>;

            fn into_iter(self) -> Self::IntoIter {
                NdIter {
                    shape: self.shape.clone(),
                    strides: self.strides,
                    data: &self.data.as_ref()[self.offset..],
                    idx: (..self.shape).first(),
                }
            }
        }
    };
}

macro_rules! impl_view_mut_methods {
    ($struct:ident $(<$lt:lifetime>)?) => {
        impl<S: Index, E, D: AsRef<[E]> + AsMut<[E]>> $struct<$($lt,)? S, E, D> {
            pub unsafe fn get_unchecked_mut(&mut self, idx: S::Dyn) -> &mut E {
                self.data.as_mut().get_unchecked_mut(self.offset + idx.to_i(&self.strides))
            }
        }

        impl<S: Index, E, D: AsRef<[E]> + AsMut<[E]>> ops::IndexMut<S::Dyn> for $struct<$($lt,)? S, E, D> {
            fn index_mut(&mut self, idx: S::Dyn) -> &mut E {
                self.shape.out_of_bounds_fail(&idx);
                unsafe { self.get_unchecked_mut(idx) }
            }
        }

        impl<'a, S: Index, E, D: AsRef<[E]> + AsMut<[E]>> IntoIterator for &'a mut $struct<$($lt,)? S, E, D> {
            type Item = &'a mut E;
            type IntoIter = NdIterMut<'a, S, E>;

            fn into_iter(self) -> Self::IntoIter {
                NdIterMut {
                    shape: self.shape.clone(),
                    strides: self.strides,
                    data: &mut self.data.as_mut()[self.offset..],
                    idx: (..self.shape).first(),
                }
            }
        }
    };
}

impl_view_methods!(View<'_>);
impl_view_methods!(ViewMut<'_>);
impl_view_methods!(Array);

impl_view_mut_methods!(ViewMut<'_>);
impl_view_mut_methods!(Array);

/////////////////////////////////////////////

pub struct NdIter<'a, S: Index, E> {
    shape: S,
    strides: S::Dyn,
    data: &'a [E],
    idx: Option<S::Dyn>,
}

impl<'a, S: Index, E> NdIter<'a, S, E> {
    pub fn nd_enumerate(self) -> NdEnumerate<Self> {
        NdEnumerate(self)
    }
}

impl<'a, S: Index, E> Iterator for NdIter<'a, S, E> {
    type Item = &'a E;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.idx?;

        let val = unsafe { self.data.get_unchecked(idx.to_i(&self.strides)) };
        self.idx = (..self.shape).next(idx);
        Some(val)
    }
}

pub struct NdIterMut<'a, S: Index, E> {
    shape: S,
    strides: S::Dyn,
    data: &'a mut [E],
    idx: Option<S::Dyn>,
}

impl<'a, S: Index, E> NdIterMut<'a, S, E> {
    pub fn nd_enumerate(self) -> NdEnumerate<Self> {
        NdEnumerate(self)
    }
}

impl<'a, S: Index, E> Iterator for NdIterMut<'a, S, E> {
    type Item = &'a mut E;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.idx?;
        let val = unsafe { &mut *(self.data.get_unchecked_mut(idx.to_i(&self.strides)) as *mut E) };
        self.idx = (..self.shape).next(idx);
        Some(val)
    }
}

pub struct NdEnumerate<I>(I);

impl<'a, S: Index, E> Iterator for NdEnumerate<NdIter<'a, S, E>> {
    type Item = (S::Dyn, &'a E);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.0.idx?;
        let val = unsafe { self.0.data.get_unchecked(idx.to_i(&self.0.strides)) };
        self.0.idx = (..self.0.shape).next(idx);
        Some((idx, val))
    }
}

impl<'a, S: Index, E> Iterator for NdEnumerate<NdIterMut<'a, S, E>> {
    type Item = (S::Dyn, &'a mut E);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.0.idx?;
        let val =
            unsafe { &mut *(self.0.data.get_unchecked_mut(idx.to_i(&self.0.strides)) as *mut E) };
        self.0.idx = (..self.0.shape).next(idx);
        Some((idx, val))
    }
}

/////////////////////////////////////////////

pub trait IntoTarget {
    type Shape: Index;
    type Element;
    type Data;

    type Output;
    type Target: OutTarget<
        Shape = Self::Shape,
        Element = Self::Element,
        Data = Self::Data,
        Output = Self::Output,
    >;

    fn build(self, shape: &Self::Shape) -> Self::Target;
}

pub trait OutTarget: IntoView + IntoViewMut {
    type Output;
    fn output(self) -> Self::Output;
}

impl<'a, S: Index, E, D> IntoTarget for &'a mut Array<S, E, D> {
    type Shape = S;
    type Element = E;
    type Data = D;

    type Target = ViewMut<'a, S, E, D>;
    type Output = ();

    fn build(self, shape: &Self::Shape) -> Self::Target {
        self.shape.shape_mismatch_fail(shape);
        self.view_mut()
    }
}

impl<'a, S: Index, E, D> IntoTarget for &'a mut ViewMut<'a, S, E, D> {
    type Shape = S;
    type Element = E;
    type Data = D;

    type Target = ViewMut<'a, S, E, D>;
    type Output = ();

    fn build(self, shape: &Self::Shape) -> Self::Target {
        self.shape.shape_mismatch_fail(shape);
        self.view_mut()
    }
}

impl<'a, S: Index, E, D> OutTarget for ViewMut<'a, S, E, D> {
    type Output = ();

    fn output(self) -> Self::Output {
        ()
    }
}

pub struct Alloc<S: Index, E>(marker::PhantomData<(S, E)>);

pub fn alloc<S: Index, E>() -> Alloc<S, E> {
    Alloc(marker::PhantomData)
}

impl<'a, S: Index, E: Default + Clone> IntoTarget for Alloc<S, E> {
    type Shape = S;
    type Element = E;
    type Data = Vec<E>;

    type Target = Array<Self::Shape, Self::Element, Self::Data>;
    type Output = Self::Target;

    fn build(self, shape: &Self::Shape) -> Self::Target {
        Array {
            shape: shape.clone(),
            strides: shape.default_strides(),
            element: marker::PhantomData,
            offset: 0,
            data: vec![E::default(); shape.num_elements()],
        }
    }
}

impl<S: Index, E, D> OutTarget for Array<S, E, D> {
    type Output = Self;

    fn output(self) -> Self::Output {
        self
    }
}

/////////////////////////////////////////////

pub struct RangeIter<R: DefiniteRange> {
    range: R,
    cur: Option<R::Item>,
}

impl<R: DefiniteRange> Iterator for RangeIter<R> {
    type Item = R::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.cur?;
        self.cur = self.range.next(cur);
        Some(cur)
    }
}

/////////////////////////////////////////////

#[cfg(test)]
mod test {

    use crate::{
        alloc, Array, Const, DefiniteRange, Index, IntoTarget, IntoView, IntoViewMut, OutTarget,
    };
    use std::marker::PhantomData;

    #[test]
    fn test() {
        let mut t = Array {
            shape: (Const::<2>, Const::<2>),
            strides: (Const::<2>, Const::<2>).default_strides(),
            element: PhantomData::<i32>,
            offset: 0,
            data: vec![1, 2, 3, 4],
        };

        for e in &t {
            dbg!(e);
        }
        for e in &mut t {
            *e *= 2;
        }
        for e in &t {
            dbg!(e);
        }
    }

    #[test]
    fn test_bernstein() {
        /// Compute the binomial coefficient (n choose k)
        fn binomial(n: usize, k: usize) -> usize {
            let numerator: usize = (1..=k).map(|i| n + 1 - i).product();
            let denominator: usize = (1..=k).product();
            numerator / denominator
        }

        fn bernstein_coef<
            S: Index,
            O: IntoTarget<Shape = S, Element = f32, Data: AsRef<[f32]> + AsMut<[f32]>>,
        >(
            c_m: &impl IntoView<Shape = S, Element = f32, Data: AsRef<[f32]>>,
            out: O,
        ) -> O::Output {
            let c_m = c_m.view();

            let mut target = out.build(&c_m.shape);
            let mut out_view = target.view_mut();

            // XXX
            //let val1 = c_m[<<V as IntoView>::Shape as Shape>::Index::zero()];

            for (i, out_entry) in (&mut out_view).into_iter().nd_enumerate() {
                *out_entry = (..=i)
                    .nd_iter()
                    .map(|j| {
                        let num: usize = i
                            .axis_iter()
                            .zip(j.axis_iter())
                            .map(|(i_n, j_n)| binomial(i_n, j_n))
                            .product();
                        let den: usize = c_m
                            .shape
                            .axis_iter()
                            .zip(j.axis_iter())
                            .map(|(d_n, j_n)| binomial(d_n, j_n))
                            .product();
                        let b = (num as f32) / (den as f32);
                        b * c_m[j]
                    })
                    .sum();
            }

            target.output()
        }

        // TEST DATA

        let a = Array {
            shape: (Const::<2>, Const::<2>),
            strides: (Const::<2>, Const::<2>).default_strides(),
            element: PhantomData,
            offset: 0,
            data: vec![1., 2., 3., 4.],
        };

        /*
        let mut b = Array {
            shape: (Const::<2>, Const::<2>),
            strides: (Const::<2>, Const::<2>).default_strides(),
            element: PhantomData,
            offset: 0,
            data: vec![0.; 4],
        };
        */

        //bernstein_coef(&a, &mut b.view_mut());
        //bernstein_coef(&a.view(), &mut b);
        dbg!(bernstein_coef(&a.view(), alloc()));
        panic!();
    }
}
