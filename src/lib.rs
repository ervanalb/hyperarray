use std::fmt;
use std::marker; //::PhantomData;
use std::ops;

/// Represents a single dimension of a multi dimensional shape
pub trait Dim: 'static + Sized + Clone + Copy + fmt::Debug {
    fn value(&self) -> usize;
}

/// Represents a dimension whose length is known at compile time
#[derive(Default, Clone, Copy)]
pub struct Const<const N: usize>;

impl<const N: usize> fmt::Debug for Const<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Const<{}>", N)
    }
}

impl<const M: usize> Dim for Const<M> {
    fn value(&self) -> usize {
        M
    }
}

impl Dim for usize {
    fn value(&self) -> usize {
        *self
    }
}

/// The shape of an ndarray, e.g. a 2D statically-sized array might have shape (Const<3>, Const<2>)
/// or a 3D dynamically-sized array might have shape (3, 4, 5)
pub trait Shape: 'static + Sized + Clone + fmt::Debug {
    /// This type is [usize; Self::NUM_DIMS] but rust doesn't allow that :(
    type Index: Index;

    // Required methods:

    fn value(&self) -> Self::Index;

    // Provided methods:

    fn num_elements(&self) -> usize {
        self.value().into_iter().product()
    }

    fn default_strides(&self) -> Self::Index {
        let mut result = Self::Index::zero();
        let mut acc = 1;
        for (i, d) in self.value().into_iter().enumerate().rev() {
            result[i] = acc;
            acc *= d;
        }
        result
    }

    fn shape_mismatch_fail(&self, other: &Self) {
        let expected = self.value();
        let got = other.value();
        if expected != got {
            panic!("Shapes do not match: expected={expected:?} got={got:?}");
        }
    }
}

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

impl<const N: usize> Index for [usize; N] {
    fn zero() -> Self {
        [0; N]
    }

    fn iter_mut<'a>(&'a mut self) -> std::slice::IterMut<'a, usize> {
        <[usize]>::iter_mut(self)
    }
}

pub trait DefiniteRange: Sized {
    type Index: Index;

    fn first(&self) -> Option<Self::Index>;
    fn next(&self, cur: Self::Index) -> Option<Self::Index>;
    fn into_iter(self) -> RangeIter<Self> {
        let cur = self.first();
        RangeIter { range: self, cur }
    }
}

impl<I: Index> DefiniteRange for ops::RangeTo<I> {
    type Index = I;

    fn first(&self) -> Option<I> {
        self.end.into_iter().all(|n| n > 0).then_some(I::zero())
    }

    fn next(&self, mut cur: I) -> Option<I> {
        for (i, n) in cur.iter_mut().rev().zip(self.end.into_iter().rev()) {
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
    type Index = I;

    fn first(&self) -> Option<I> {
        Some(I::zero())
    }

    fn next(&self, mut cur: I) -> Option<I> {
        for (i, n) in cur.iter_mut().rev().zip(self.end.into_iter().rev()) {
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
    type Index = I;

    fn first(&self) -> Option<I> {
        self.start
            .into_iter()
            .zip(self.end.into_iter())
            .all(|(s, e)| e > s)
            .then_some(self.start)
    }

    fn next(&self, mut cur: I) -> Option<I> {
        for (i, (s, e)) in cur
            .iter_mut()
            .rev()
            .zip(self.start.into_iter().rev().zip(self.end.into_iter().rev()))
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
    type Index = I;

    fn first(&self) -> Option<I> {
        self.start()
            .into_iter()
            .zip(self.end().into_iter())
            .all(|(s, e)| e >= s)
            .then_some(*self.start())
    }

    fn next(&self, mut cur: I) -> Option<I> {
        for (i, (s, e)) in cur.iter_mut().rev().zip(
            self.start()
                .into_iter()
                .rev()
                .zip(self.end().into_iter().rev()),
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

impl<D1: Dim> Shape for (D1,) {
    type Index = [usize; 1];

    fn value(&self) -> Self::Index {
        [self.0.value()]
    }
}

impl<D1: Dim, D2: Dim> Shape for (D1, D2) {
    type Index = [usize; 2];

    fn value(&self) -> Self::Index {
        [self.0.value(), self.1.value()]
    }
}

/////////////////////////////////////////////

#[derive(Debug)]
pub struct ViewParameters<S: Shape, E> {
    shape: S,
    element: marker::PhantomData<E>,
    offset: usize,
    strides: S::Index,
}

impl<S: Shape, E> Clone for ViewParameters<S, E> {
    fn clone(&self) -> ViewParameters<S, E> {
        ViewParameters {
            shape: self.shape.clone(),
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides,
        }
    }
}

// Wrappers for parameters + data
// Three kinds: owned, ref, mut

#[derive(Debug, Clone)]
pub struct Array<S: Shape, E, D> {
    parameters: ViewParameters<S, E>,
    data: D,
}

#[derive(Debug)]
pub struct View<'a, S: Shape, E, D> {
    parameters: ViewParameters<S, E>,
    data: &'a D,
}

#[derive(Debug)]
pub struct ViewMut<'a, S: Shape, E, D> {
    parameters: ViewParameters<S, E>,
    data: &'a mut D,
}

// Get a view
pub trait IntoView {
    type Shape: Shape;
    type Element;
    type Data;

    fn parameters(&self) -> ViewParameters<Self::Shape, Self::Element>;
    fn data(&self) -> &Self::Data;

    fn view(&self) -> View<Self::Shape, Self::Element, Self::Data> {
        View {
            parameters: self.parameters(),
            data: self.data(),
        }
    }
}

impl<S: Shape, E, D> IntoView for Array<S, E, D> {
    type Shape = S;
    type Element = E;
    type Data = D;

    fn parameters(&self) -> ViewParameters<Self::Shape, Self::Element> {
        self.parameters.clone()
    }

    fn data(&self) -> &D {
        &self.data
    }
}

impl<S: Shape, E, D> IntoView for View<'_, S, E, D> {
    type Shape = S;
    type Element = E;
    type Data = D;

    fn parameters(&self) -> ViewParameters<Self::Shape, Self::Element> {
        self.parameters.clone()
    }

    fn data(&self) -> &D {
        self.data
    }
}

impl<S: Shape, E, D> IntoView for ViewMut<'_, S, E, D> {
    type Shape = S;
    type Element = E;
    type Data = D;

    fn parameters(&self) -> ViewParameters<Self::Shape, Self::Element> {
        self.parameters.clone()
    }

    fn data(&self) -> &D {
        self.data
    }
}

// Get a mutable view
pub trait IntoViewMut: IntoView {
    fn data_mut(&mut self) -> &mut Self::Data;

    fn view_mut(&mut self) -> ViewMut<'_, Self::Shape, Self::Element, Self::Data> {
        ViewMut {
            parameters: self.parameters(),
            data: self.data_mut(),
        }
    }
}

impl<S: Shape, E, D> IntoViewMut for Array<S, E, D> {
    fn data_mut(&mut self) -> &mut D {
        &mut self.data
    }
}

impl<S: Shape, E, D> IntoViewMut for ViewMut<'_, S, E, D> {
    fn data_mut(&mut self) -> &mut D {
        self.data
    }
}

/////////////////////////////////////////////

impl<S: Shape, E> ViewParameters<S, E> {
    unsafe fn get_unchecked<'a>(&self, data: &'a [E], index: S::Index) -> &'a E {
        data.get_unchecked(self.offset + index.to_i(&self.strides))
    }

    unsafe fn get_unchecked_mut<'a>(&mut self, data: &'a mut [E], index: S::Index) -> &'a mut E {
        data.get_unchecked_mut(self.offset + index.to_i(&self.strides))
    }

    fn index<'a>(&self, data: &'a [E], index: S::Index) -> &'a E {
        index.out_of_bounds_fail(&self.shape);
        unsafe { self.get_unchecked(data, index) }
    }

    fn index_mut<'a>(&mut self, data: &'a mut [E], index: S::Index) -> &'a mut E {
        index.out_of_bounds_fail(&self.shape);
        unsafe { self.get_unchecked_mut(data, index) }
    }
}

macro_rules! impl_view_methods {
    ($struct:ident $(<$lt:lifetime>)?) => {
        impl<S: Shape, E, D: AsRef<[E]>> $struct<$($lt,)? S, E, D> {
            pub unsafe fn get_unchecked(&self, idx: S::Index) -> &E {
                self.parameters().get_unchecked(self.data().as_ref(), idx)
            }
        }

        impl<S: Shape, E, D: AsRef<[E]>> ops::Index<S::Index> for $struct<$($lt,)? S, E, D> {
            type Output = E;

            fn index(&self, idx: S::Index) -> &E {
                self.parameters().index(self.data().as_ref(), idx)
            }
        }

        impl<'a, S: Shape, E, D: AsRef<[E]>> IntoIterator for &'a $struct<$($lt,)? S, E, D> {
            type Item = &'a E;
            type IntoIter = NdIter<'a, S, E>;

            fn into_iter(self) -> Self::IntoIter {
                NdIter::new(self.parameters(), self.data().as_ref())
            }
        }
    };
}

macro_rules! impl_view_mut_methods {
    ($struct:ident $(<$lt:lifetime>)?) => {
        impl<S: Shape, E, D: AsRef<[E]> + AsMut<[E]>> $struct<$($lt,)? S, E, D> {
            pub unsafe fn get_unchecked_mut(&mut self, idx: S::Index) -> &E {
                self.parameters().get_unchecked_mut(self.data_mut().as_mut(), idx)
            }
        }

        impl<S: Shape, E, D: AsRef<[E]> + AsMut<[E]>> ops::IndexMut<S::Index> for $struct<$($lt,)? S, E, D> {
            fn index_mut(&mut self, idx: S::Index) -> &mut E {
                self.parameters().index_mut(self.data_mut().as_mut(), idx)
            }
        }

        impl<'a, S: Shape, E, D: AsRef<[E]> + AsMut<[E]>> IntoIterator for &'a mut $struct<$($lt,)? S, E, D> {
            type Item = &'a mut E;
            type IntoIter = NdIterMut<'a, S, E>;

            fn into_iter(self) -> Self::IntoIter {
                NdIterMut::new(self.parameters(), self.data_mut().as_mut())
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

pub struct NdIter<'a, S: Shape, E> {
    parameters: ViewParameters<S, E>,
    data: &'a [E],
    idx: Option<S::Index>,
}

impl<'a, S: Shape, E> NdIter<'a, S, E> {
    fn new(parameters: ViewParameters<S, E>, data: &'a [E]) -> Self {
        let idx = (parameters.shape.num_elements() > 0).then_some(S::Index::zero());
        NdIter {
            parameters,
            data,
            idx,
        }
    }

    pub fn nd_enumerate(self) -> NdEnumerate<Self> {
        NdEnumerate(self)
    }
}

impl<'a, S: Shape, E> Iterator for NdIter<'a, S, E> {
    type Item = &'a E;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.idx?;
        let val = unsafe { self.parameters.get_unchecked(self.data, idx) };
        self.idx = (..self.parameters.shape.value()).next(idx);
        Some(val)
    }
}

pub struct NdIterMut<'a, S: Shape, E> {
    parameters: ViewParameters<S, E>,
    data: &'a mut [E],
    idx: Option<S::Index>,
}

impl<'a, S: Shape, E> NdIterMut<'a, S, E> {
    fn new(parameters: ViewParameters<S, E>, data: &'a mut [E]) -> Self {
        let idx = (parameters.shape.num_elements() > 0).then_some(S::Index::zero());
        NdIterMut {
            parameters,
            data,
            idx,
        }
    }

    pub fn nd_enumerate(self) -> NdEnumerate<Self> {
        NdEnumerate(self)
    }
}

impl<'a, S: Shape, E> Iterator for NdIterMut<'a, S, E> {
    type Item = &'a mut E;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.idx?;
        let val = unsafe { &mut *(self.parameters.get_unchecked_mut(self.data, idx) as *mut E) };
        self.idx = (..self.parameters.shape.value()).next(idx);
        Some(val)
    }
}

pub struct NdEnumerate<I>(I);

impl<'a, S: Shape, E> Iterator for NdEnumerate<NdIter<'a, S, E>> {
    type Item = (S::Index, &'a E);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.0.idx?;
        let val = unsafe { self.0.parameters.get_unchecked(self.0.data, idx) };
        self.0.idx = (..self.0.parameters.shape.value()).next(idx);
        Some((idx, val))
    }
}

impl<'a, S: Shape, E> Iterator for NdEnumerate<NdIterMut<'a, S, E>> {
    type Item = (S::Index, &'a mut E);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.0.idx?;
        let val =
            unsafe { &mut *(self.0.parameters.get_unchecked_mut(self.0.data, idx) as *mut E) };
        self.0.idx = (..self.0.parameters.shape.value()).next(idx);
        Some((idx, val))
    }
}

/////////////////////////////////////////////

pub trait IntoTarget {
    type Shape: Shape;
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

impl<'a, S: Shape, E, D> IntoTarget for &'a mut Array<S, E, D> {
    type Shape = S;
    type Element = E;
    type Data = D;

    type Target = ViewMut<'a, S, E, D>;
    type Output = ();

    fn build(self, shape: &Self::Shape) -> Self::Target {
        self.parameters.shape.shape_mismatch_fail(shape);
        self.view_mut()
    }
}

impl<'a, S: Shape, E, D> IntoTarget for &'a mut ViewMut<'a, S, E, D> {
    type Shape = S;
    type Element = E;
    type Data = D;

    type Target = ViewMut<'a, S, E, D>;
    type Output = ();

    fn build(self, shape: &Self::Shape) -> Self::Target {
        self.parameters.shape.shape_mismatch_fail(shape);
        self.view_mut()
    }
}

impl<'a, S: Shape, E, D> OutTarget for ViewMut<'a, S, E, D> {
    type Output = ();

    fn output(self) -> Self::Output {
        ()
    }
}

pub struct Alloc<S: Shape, E>(marker::PhantomData<(S, E)>);

pub fn alloc<S: Shape, E>() -> Alloc<S, E> {
    Alloc(marker::PhantomData)
}

impl<'a, S: Shape, E: Default + Clone> IntoTarget for Alloc<S, E> {
    type Shape = S;
    type Element = E;
    type Data = Vec<E>;

    type Target = Array<Self::Shape, Self::Element, Self::Data>;
    type Output = Self::Target;

    fn build(self, shape: &Self::Shape) -> Self::Target {
        Array {
            parameters: ViewParameters {
                shape: shape.clone(),
                strides: shape.default_strides(),
                element: marker::PhantomData,
                offset: 0,
            },
            data: vec![E::default(); shape.num_elements()],
        }
    }
}

impl<S: Shape, E, D> OutTarget for Array<S, E, D> {
    type Output = Self;

    fn output(self) -> Self::Output {
        self
    }
}

/////////////////////////////////////////////

pub struct RangeIter<R: DefiniteRange> {
    range: R,
    cur: Option<R::Index>,
}

impl<R: DefiniteRange> Iterator for RangeIter<R> {
    type Item = R::Index;

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
        alloc, Array, Const, DefiniteRange, IntoTarget, IntoView, IntoViewMut, OutTarget, Shape,
        ViewParameters,
    };
    use std::marker::PhantomData;

    #[test]
    fn test() {
        let mut t = Array {
            parameters: ViewParameters {
                shape: (Const::<2>, Const::<2>),
                strides: (Const::<2>, Const::<2>).default_strides(),
                element: PhantomData::<i32>,
                offset: 0,
            },
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
            S: Shape,
            O: IntoTarget<Shape = S, Element = f32, Data: AsRef<[f32]> + AsMut<[f32]>>,
        >(
            c_m: &impl IntoView<Shape = S, Element = f32, Data: AsRef<[f32]>>,
            out: O,
        ) -> O::Output {
            let c_m = c_m.view();

            let mut target = out.build(&c_m.parameters.shape);
            let mut out_view = target.view_mut();

            // XXX
            //let val1 = c_m[<<V as IntoView>::Shape as Shape>::Index::zero()];

            for (i, out_entry) in (&mut out_view).into_iter().nd_enumerate() {
                *out_entry = (..=i)
                    .into_iter()
                    .map(|j| {
                        let num: usize = i
                            .into_iter()
                            .zip(j.into_iter())
                            .map(|(i_n, j_n)| binomial(i_n, j_n))
                            .product();
                        let den: usize = c_m
                            .parameters
                            .shape
                            .value()
                            .into_iter()
                            .zip(j.into_iter())
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
            parameters: ViewParameters {
                shape: (Const::<2>, Const::<2>),
                strides: (Const::<2>, Const::<2>).default_strides(),
                element: PhantomData,
                offset: 0,
            },
            data: vec![1., 2., 3., 4.],
        };

        /*
        let mut b = Array {
            parameters: ViewParameters {
                shape: (Const::<2>, Const::<2>),
                strides: (Const::<2>, Const::<2>).default_strides(),
                element: PhantomData,
                offset: 0,
            },
            data: vec![0.; 4],
        };
        */

        //bernstein_coef(&a, &mut b.view_mut());
        //bernstein_coef(&a.view(), &mut b);
        dbg!(bernstein_coef(&a.view(), alloc()));
        panic!();
    }
}
