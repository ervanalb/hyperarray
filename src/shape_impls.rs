
use crate::{AsIndex, Const, NewAxis, Shape, ShapeEq, IntoIndex, BroadcastShape};


impl<const N0: usize, const N1: usize, > AsIndex for (Const<N1>, Const<N0>, )
{
    type Index = [usize; 2];

    fn as_index(&self) -> Self::Index {
        [N1, N0, ]
    }
}
impl<const N0: usize, const N1: usize, > Shape for (Const<N1>, Const<N0>, ) {}
    

impl<const N0: usize, > AsIndex for (usize, Const<N0>, )
{
    type Index = [usize; 2];

    fn as_index(&self) -> Self::Index {
        [self.0, N0, ]
    }
}
impl<const N0: usize, > Shape for (usize, Const<N0>, ) {}
    

impl<const N0: usize, > AsIndex for (Const<N0>, )
{
    type Index = [usize; 1];

    fn as_index(&self) -> Self::Index {
        [N0, ]
    }
}
impl<const N0: usize, > Shape for (Const<N0>, ) {}
    

impl<const N1: usize, > AsIndex for (Const<N1>, usize, )
{
    type Index = [usize; 2];

    fn as_index(&self) -> Self::Index {
        [N1, self.1, ]
    }
}
impl<const N1: usize, > Shape for (Const<N1>, usize, ) {}
    

impl AsIndex for (usize, usize, )
{
    type Index = [usize; 2];

    fn as_index(&self) -> Self::Index {
        [self.0, self.1, ]
    }
}
impl Shape for (usize, usize, ) {}
    

impl AsIndex for (usize, )
{
    type Index = [usize; 1];

    fn as_index(&self) -> Self::Index {
        [self.0, ]
    }
}
impl Shape for (usize, ) {}
    

impl<const N1: usize, > AsIndex for (Const<N1>, NewAxis, )
{
    type Index = [usize; 1];

    fn as_index(&self) -> Self::Index {
        [N1, ]
    }
}
impl<const N1: usize, > Shape for (Const<N1>, NewAxis, ) {}
    

impl AsIndex for (usize, NewAxis, )
{
    type Index = [usize; 1];

    fn as_index(&self) -> Self::Index {
        [self.0, ]
    }
}
impl Shape for (usize, NewAxis, ) {}
    

impl AsIndex for ()
{
    type Index = [usize; 0];

    fn as_index(&self) -> Self::Index {
        []
    }
}
impl Shape for () {}
    
//////////////////////////////

impl<const N0: usize, const N1: usize, > ShapeEq<(Const<N1>, Const<N0>, )> for (Const<N1>, Const<N0>, )
{
    fn shape_eq(&self, _other: &(Const<N1>, Const<N0>, )) -> bool {
        true
    }
}


impl<const N0: usize, const N1: usize, > IntoIndex<(Const<N1>, Const<N0>, )> for (Const<N1>, Const<N0>, )
{
    fn into_index(index: [usize; 2]) -> [usize; 2] {
        [index[1], index[0], ]
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N1>, Const<N0>, )> for (Const<N1>, Const<N0>, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N1>, Const<N0>, )) -> Option<Self::Output> {
        Some((Const, Const, ))
    }
}


impl<const N0: usize, const N1: usize, > ShapeEq<(usize, Const<N0>, )> for (Const<N1>, Const<N0>, )
{
    fn shape_eq(&self, other: &(usize, Const<N0>, )) -> bool {
        N1 == other.0
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(usize, Const<N0>, )> for (Const<N1>, Const<N0>, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, other: (usize, Const<N0>, )) -> Option<Self::Output> {
        Some(((N1 == other.0).then_some(Const)?, Const, ))
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N0>, )> for (Const<N1>, Const<N0>, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N0>, )) -> Option<Self::Output> {
        Some((Const, Const, ))
    }
}


impl<const N0: usize, const N1: usize, > ShapeEq<(Const<N1>, usize, )> for (Const<N1>, Const<N0>, )
{
    fn shape_eq(&self, other: &(Const<N1>, usize, )) -> bool {
        N0 == other.1
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N1>, usize, )> for (Const<N1>, Const<N0>, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, other: (Const<N1>, usize, )) -> Option<Self::Output> {
        Some((Const, (N0 == other.1).then_some(Const)?, ))
    }
}


impl<const N0: usize, const N1: usize, > ShapeEq<(usize, usize, )> for (Const<N1>, Const<N0>, )
{
    fn shape_eq(&self, other: &(usize, usize, )) -> bool {
        N1 == other.0 && N0 == other.1
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(usize, usize, )> for (Const<N1>, Const<N0>, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, other: (usize, usize, )) -> Option<Self::Output> {
        Some(((N1 == other.0).then_some(Const)?, (N0 == other.1).then_some(Const)?, ))
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(usize, )> for (Const<N1>, Const<N0>, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, other: (usize, )) -> Option<Self::Output> {
        Some((Const, (N0 == other.0).then_some(Const)?, ))
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N1>, NewAxis, )> for (Const<N1>, Const<N0>, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N1>, NewAxis, )) -> Option<Self::Output> {
        Some((Const, Const, ))
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(usize, NewAxis, )> for (Const<N1>, Const<N0>, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, other: (usize, NewAxis, )) -> Option<Self::Output> {
        Some(((N1 == other.0).then_some(Const)?, Const, ))
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<()> for (Const<N1>, Const<N0>, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, _other: ()) -> Option<Self::Output> {
        Some((Const, Const, ))
    }
}


impl<const N0: usize, const N1: usize, > ShapeEq<(Const<N1>, Const<N0>, )> for (usize, Const<N0>, )
{
    fn shape_eq(&self, _other: &(Const<N1>, Const<N0>, )) -> bool {
        self.0 == N1
    }
}


impl<const N0: usize, const N1: usize, > IntoIndex<(Const<N1>, Const<N0>, )> for (usize, Const<N0>, )
{
    fn into_index(index: [usize; 2]) -> [usize; 2] {
        [index[1], index[0], ]
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N1>, Const<N0>, )> for (usize, Const<N0>, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N1>, Const<N0>, )) -> Option<Self::Output> {
        Some(((self.0 == N1).then_some(Const)?, Const, ))
    }
}


impl<const N0: usize, > ShapeEq<(usize, Const<N0>, )> for (usize, Const<N0>, )
{
    fn shape_eq(&self, other: &(usize, Const<N0>, )) -> bool {
        self.0 == other.0
    }
}


impl<const N0: usize, > IntoIndex<(usize, Const<N0>, )> for (usize, Const<N0>, )
{
    fn into_index(index: [usize; 2]) -> [usize; 2] {
        [index[1], index[0], ]
    }
}


impl<const N0: usize, > BroadcastShape<(usize, Const<N0>, )> for (usize, Const<N0>, )
{
    type Output = (usize, Const<N0>, );
    fn broadcast_shape(self, other: (usize, Const<N0>, )) -> Option<Self::Output> {
        Some(((self.0 == other.0).then_some(self.0)?, Const, ))
    }
}


impl<const N0: usize, > BroadcastShape<(Const<N0>, )> for (usize, Const<N0>, )
{
    type Output = (usize, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N0>, )) -> Option<Self::Output> {
        Some((self.0, Const, ))
    }
}


impl<const N0: usize, const N1: usize, > ShapeEq<(Const<N1>, usize, )> for (usize, Const<N0>, )
{
    fn shape_eq(&self, other: &(Const<N1>, usize, )) -> bool {
        self.0 == N1 && N0 == other.1
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N1>, usize, )> for (usize, Const<N0>, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, other: (Const<N1>, usize, )) -> Option<Self::Output> {
        Some(((self.0 == N1).then_some(Const)?, (N0 == other.1).then_some(Const)?, ))
    }
}


impl<const N0: usize, > ShapeEq<(usize, usize, )> for (usize, Const<N0>, )
{
    fn shape_eq(&self, other: &(usize, usize, )) -> bool {
        self.0 == other.0 && N0 == other.1
    }
}


impl<const N0: usize, > BroadcastShape<(usize, usize, )> for (usize, Const<N0>, )
{
    type Output = (usize, Const<N0>, );
    fn broadcast_shape(self, other: (usize, usize, )) -> Option<Self::Output> {
        Some(((self.0 == other.0).then_some(self.0)?, (N0 == other.1).then_some(Const)?, ))
    }
}


impl<const N0: usize, > BroadcastShape<(usize, )> for (usize, Const<N0>, )
{
    type Output = (usize, Const<N0>, );
    fn broadcast_shape(self, other: (usize, )) -> Option<Self::Output> {
        Some((self.0, (N0 == other.0).then_some(Const)?, ))
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N1>, NewAxis, )> for (usize, Const<N0>, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N1>, NewAxis, )) -> Option<Self::Output> {
        Some(((self.0 == N1).then_some(Const)?, Const, ))
    }
}


impl<const N0: usize, > BroadcastShape<(usize, NewAxis, )> for (usize, Const<N0>, )
{
    type Output = (usize, Const<N0>, );
    fn broadcast_shape(self, other: (usize, NewAxis, )) -> Option<Self::Output> {
        Some(((self.0 == other.0).then_some(self.0)?, Const, ))
    }
}


impl<const N0: usize, > BroadcastShape<()> for (usize, Const<N0>, )
{
    type Output = (usize, Const<N0>, );
    fn broadcast_shape(self, _other: ()) -> Option<Self::Output> {
        Some((self.0, Const, ))
    }
}


impl<const N0: usize, const N1: usize, > IntoIndex<(Const<N1>, Const<N0>, )> for (Const<N0>, )
{
    fn into_index(index: [usize; 1]) -> [usize; 2] {
        [index[0], 0, ]
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N1>, Const<N0>, )> for (Const<N0>, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N1>, Const<N0>, )) -> Option<Self::Output> {
        Some((Const, Const, ))
    }
}


impl<const N0: usize, > IntoIndex<(usize, Const<N0>, )> for (Const<N0>, )
{
    fn into_index(index: [usize; 1]) -> [usize; 2] {
        [index[0], 0, ]
    }
}


impl<const N0: usize, > BroadcastShape<(usize, Const<N0>, )> for (Const<N0>, )
{
    type Output = (usize, Const<N0>, );
    fn broadcast_shape(self, other: (usize, Const<N0>, )) -> Option<Self::Output> {
        Some((other.0, Const, ))
    }
}


impl<const N0: usize, > ShapeEq<(Const<N0>, )> for (Const<N0>, )
{
    fn shape_eq(&self, _other: &(Const<N0>, )) -> bool {
        true
    }
}


impl<const N0: usize, > IntoIndex<(Const<N0>, )> for (Const<N0>, )
{
    fn into_index(index: [usize; 1]) -> [usize; 1] {
        [index[0], ]
    }
}


impl<const N0: usize, > BroadcastShape<(Const<N0>, )> for (Const<N0>, )
{
    type Output = (Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N0>, )) -> Option<Self::Output> {
        Some((Const, ))
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N1>, usize, )> for (Const<N0>, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, other: (Const<N1>, usize, )) -> Option<Self::Output> {
        Some((Const, (N0 == other.1).then_some(Const)?, ))
    }
}


impl<const N0: usize, > BroadcastShape<(usize, usize, )> for (Const<N0>, )
{
    type Output = (usize, Const<N0>, );
    fn broadcast_shape(self, other: (usize, usize, )) -> Option<Self::Output> {
        Some((other.0, (N0 == other.1).then_some(Const)?, ))
    }
}


impl<const N0: usize, > ShapeEq<(usize, )> for (Const<N0>, )
{
    fn shape_eq(&self, other: &(usize, )) -> bool {
        N0 == other.0
    }
}


impl<const N0: usize, > BroadcastShape<(usize, )> for (Const<N0>, )
{
    type Output = (Const<N0>, );
    fn broadcast_shape(self, other: (usize, )) -> Option<Self::Output> {
        Some(((N0 == other.0).then_some(Const)?, ))
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N1>, NewAxis, )> for (Const<N0>, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N1>, NewAxis, )) -> Option<Self::Output> {
        Some((Const, Const, ))
    }
}


impl<const N0: usize, > BroadcastShape<(usize, NewAxis, )> for (Const<N0>, )
{
    type Output = (usize, Const<N0>, );
    fn broadcast_shape(self, other: (usize, NewAxis, )) -> Option<Self::Output> {
        Some((other.0, Const, ))
    }
}


impl<const N0: usize, > BroadcastShape<()> for (Const<N0>, )
{
    type Output = (Const<N0>, );
    fn broadcast_shape(self, _other: ()) -> Option<Self::Output> {
        Some((Const, ))
    }
}


impl<const N0: usize, const N1: usize, > ShapeEq<(Const<N1>, Const<N0>, )> for (Const<N1>, usize, )
{
    fn shape_eq(&self, _other: &(Const<N1>, Const<N0>, )) -> bool {
        self.1 == N0
    }
}


impl<const N0: usize, const N1: usize, > IntoIndex<(Const<N1>, Const<N0>, )> for (Const<N1>, usize, )
{
    fn into_index(index: [usize; 2]) -> [usize; 2] {
        [index[1], index[0], ]
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N1>, Const<N0>, )> for (Const<N1>, usize, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N1>, Const<N0>, )) -> Option<Self::Output> {
        Some((Const, (self.1 == N0).then_some(Const)?, ))
    }
}


impl<const N0: usize, const N1: usize, > ShapeEq<(usize, Const<N0>, )> for (Const<N1>, usize, )
{
    fn shape_eq(&self, other: &(usize, Const<N0>, )) -> bool {
        N1 == other.0 && self.1 == N0
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(usize, Const<N0>, )> for (Const<N1>, usize, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, other: (usize, Const<N0>, )) -> Option<Self::Output> {
        Some(((N1 == other.0).then_some(Const)?, (self.1 == N0).then_some(Const)?, ))
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N0>, )> for (Const<N1>, usize, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N0>, )) -> Option<Self::Output> {
        Some((Const, (self.1 == N0).then_some(Const)?, ))
    }
}


impl<const N1: usize, > ShapeEq<(Const<N1>, usize, )> for (Const<N1>, usize, )
{
    fn shape_eq(&self, other: &(Const<N1>, usize, )) -> bool {
        self.1 == other.1
    }
}


impl<const N1: usize, > IntoIndex<(Const<N1>, usize, )> for (Const<N1>, usize, )
{
    fn into_index(index: [usize; 2]) -> [usize; 2] {
        [index[1], index[0], ]
    }
}


impl<const N1: usize, > BroadcastShape<(Const<N1>, usize, )> for (Const<N1>, usize, )
{
    type Output = (Const<N1>, usize, );
    fn broadcast_shape(self, other: (Const<N1>, usize, )) -> Option<Self::Output> {
        Some((Const, (self.1 == other.1).then_some(self.1)?, ))
    }
}


impl<const N1: usize, > ShapeEq<(usize, usize, )> for (Const<N1>, usize, )
{
    fn shape_eq(&self, other: &(usize, usize, )) -> bool {
        N1 == other.0 && self.1 == other.1
    }
}


impl<const N1: usize, > BroadcastShape<(usize, usize, )> for (Const<N1>, usize, )
{
    type Output = (Const<N1>, usize, );
    fn broadcast_shape(self, other: (usize, usize, )) -> Option<Self::Output> {
        Some(((N1 == other.0).then_some(Const)?, (self.1 == other.1).then_some(self.1)?, ))
    }
}


impl<const N1: usize, > BroadcastShape<(usize, )> for (Const<N1>, usize, )
{
    type Output = (Const<N1>, usize, );
    fn broadcast_shape(self, other: (usize, )) -> Option<Self::Output> {
        Some((Const, (self.1 == other.0).then_some(self.1)?, ))
    }
}


impl<const N1: usize, > BroadcastShape<(Const<N1>, NewAxis, )> for (Const<N1>, usize, )
{
    type Output = (Const<N1>, usize, );
    fn broadcast_shape(self, _other: (Const<N1>, NewAxis, )) -> Option<Self::Output> {
        Some((Const, self.1, ))
    }
}


impl<const N1: usize, > BroadcastShape<(usize, NewAxis, )> for (Const<N1>, usize, )
{
    type Output = (Const<N1>, usize, );
    fn broadcast_shape(self, other: (usize, NewAxis, )) -> Option<Self::Output> {
        Some(((N1 == other.0).then_some(Const)?, self.1, ))
    }
}


impl<const N1: usize, > BroadcastShape<()> for (Const<N1>, usize, )
{
    type Output = (Const<N1>, usize, );
    fn broadcast_shape(self, _other: ()) -> Option<Self::Output> {
        Some((Const, self.1, ))
    }
}


impl<const N0: usize, const N1: usize, > ShapeEq<(Const<N1>, Const<N0>, )> for (usize, usize, )
{
    fn shape_eq(&self, _other: &(Const<N1>, Const<N0>, )) -> bool {
        self.0 == N1 && self.1 == N0
    }
}


impl<const N0: usize, const N1: usize, > IntoIndex<(Const<N1>, Const<N0>, )> for (usize, usize, )
{
    fn into_index(index: [usize; 2]) -> [usize; 2] {
        [index[1], index[0], ]
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N1>, Const<N0>, )> for (usize, usize, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N1>, Const<N0>, )) -> Option<Self::Output> {
        Some(((self.0 == N1).then_some(Const)?, (self.1 == N0).then_some(Const)?, ))
    }
}


impl<const N0: usize, > ShapeEq<(usize, Const<N0>, )> for (usize, usize, )
{
    fn shape_eq(&self, other: &(usize, Const<N0>, )) -> bool {
        self.0 == other.0 && self.1 == N0
    }
}


impl<const N0: usize, > IntoIndex<(usize, Const<N0>, )> for (usize, usize, )
{
    fn into_index(index: [usize; 2]) -> [usize; 2] {
        [index[1], index[0], ]
    }
}


impl<const N0: usize, > BroadcastShape<(usize, Const<N0>, )> for (usize, usize, )
{
    type Output = (usize, Const<N0>, );
    fn broadcast_shape(self, other: (usize, Const<N0>, )) -> Option<Self::Output> {
        Some(((self.0 == other.0).then_some(self.0)?, (self.1 == N0).then_some(Const)?, ))
    }
}


impl<const N0: usize, > BroadcastShape<(Const<N0>, )> for (usize, usize, )
{
    type Output = (usize, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N0>, )) -> Option<Self::Output> {
        Some((self.0, (self.1 == N0).then_some(Const)?, ))
    }
}


impl<const N1: usize, > ShapeEq<(Const<N1>, usize, )> for (usize, usize, )
{
    fn shape_eq(&self, other: &(Const<N1>, usize, )) -> bool {
        self.0 == N1 && self.1 == other.1
    }
}


impl<const N1: usize, > IntoIndex<(Const<N1>, usize, )> for (usize, usize, )
{
    fn into_index(index: [usize; 2]) -> [usize; 2] {
        [index[1], index[0], ]
    }
}


impl<const N1: usize, > BroadcastShape<(Const<N1>, usize, )> for (usize, usize, )
{
    type Output = (Const<N1>, usize, );
    fn broadcast_shape(self, other: (Const<N1>, usize, )) -> Option<Self::Output> {
        Some(((self.0 == N1).then_some(Const)?, (self.1 == other.1).then_some(self.1)?, ))
    }
}


impl ShapeEq<(usize, usize, )> for (usize, usize, )
{
    fn shape_eq(&self, other: &(usize, usize, )) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}


impl IntoIndex<(usize, usize, )> for (usize, usize, )
{
    fn into_index(index: [usize; 2]) -> [usize; 2] {
        [index[1], index[0], ]
    }
}


impl BroadcastShape<(usize, usize, )> for (usize, usize, )
{
    type Output = (usize, usize, );
    fn broadcast_shape(self, other: (usize, usize, )) -> Option<Self::Output> {
        Some(((self.0 == other.0).then_some(self.0)?, (self.1 == other.1).then_some(self.1)?, ))
    }
}


impl BroadcastShape<(usize, )> for (usize, usize, )
{
    type Output = (usize, usize, );
    fn broadcast_shape(self, other: (usize, )) -> Option<Self::Output> {
        Some((self.0, (self.1 == other.0).then_some(self.1)?, ))
    }
}


impl<const N1: usize, > BroadcastShape<(Const<N1>, NewAxis, )> for (usize, usize, )
{
    type Output = (Const<N1>, usize, );
    fn broadcast_shape(self, _other: (Const<N1>, NewAxis, )) -> Option<Self::Output> {
        Some(((self.0 == N1).then_some(Const)?, self.1, ))
    }
}


impl BroadcastShape<(usize, NewAxis, )> for (usize, usize, )
{
    type Output = (usize, usize, );
    fn broadcast_shape(self, other: (usize, NewAxis, )) -> Option<Self::Output> {
        Some(((self.0 == other.0).then_some(self.0)?, self.1, ))
    }
}


impl BroadcastShape<()> for (usize, usize, )
{
    type Output = (usize, usize, );
    fn broadcast_shape(self, _other: ()) -> Option<Self::Output> {
        Some((self.0, self.1, ))
    }
}


impl<const N0: usize, const N1: usize, > IntoIndex<(Const<N1>, Const<N0>, )> for (usize, )
{
    fn into_index(index: [usize; 1]) -> [usize; 2] {
        [index[0], 0, ]
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N1>, Const<N0>, )> for (usize, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N1>, Const<N0>, )) -> Option<Self::Output> {
        Some((Const, (self.0 == N0).then_some(Const)?, ))
    }
}


impl<const N0: usize, > IntoIndex<(usize, Const<N0>, )> for (usize, )
{
    fn into_index(index: [usize; 1]) -> [usize; 2] {
        [index[0], 0, ]
    }
}


impl<const N0: usize, > BroadcastShape<(usize, Const<N0>, )> for (usize, )
{
    type Output = (usize, Const<N0>, );
    fn broadcast_shape(self, other: (usize, Const<N0>, )) -> Option<Self::Output> {
        Some((other.0, (self.0 == N0).then_some(Const)?, ))
    }
}


impl<const N0: usize, > ShapeEq<(Const<N0>, )> for (usize, )
{
    fn shape_eq(&self, _other: &(Const<N0>, )) -> bool {
        self.0 == N0
    }
}


impl<const N0: usize, > IntoIndex<(Const<N0>, )> for (usize, )
{
    fn into_index(index: [usize; 1]) -> [usize; 1] {
        [index[0], ]
    }
}


impl<const N0: usize, > BroadcastShape<(Const<N0>, )> for (usize, )
{
    type Output = (Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N0>, )) -> Option<Self::Output> {
        Some(((self.0 == N0).then_some(Const)?, ))
    }
}


impl<const N1: usize, > IntoIndex<(Const<N1>, usize, )> for (usize, )
{
    fn into_index(index: [usize; 1]) -> [usize; 2] {
        [index[0], 0, ]
    }
}


impl<const N1: usize, > BroadcastShape<(Const<N1>, usize, )> for (usize, )
{
    type Output = (Const<N1>, usize, );
    fn broadcast_shape(self, other: (Const<N1>, usize, )) -> Option<Self::Output> {
        Some((Const, (self.0 == other.1).then_some(self.0)?, ))
    }
}


impl IntoIndex<(usize, usize, )> for (usize, )
{
    fn into_index(index: [usize; 1]) -> [usize; 2] {
        [index[0], 0, ]
    }
}


impl BroadcastShape<(usize, usize, )> for (usize, )
{
    type Output = (usize, usize, );
    fn broadcast_shape(self, other: (usize, usize, )) -> Option<Self::Output> {
        Some((other.0, (self.0 == other.1).then_some(self.0)?, ))
    }
}


impl ShapeEq<(usize, )> for (usize, )
{
    fn shape_eq(&self, other: &(usize, )) -> bool {
        self.0 == other.0
    }
}


impl IntoIndex<(usize, )> for (usize, )
{
    fn into_index(index: [usize; 1]) -> [usize; 1] {
        [index[0], ]
    }
}


impl BroadcastShape<(usize, )> for (usize, )
{
    type Output = (usize, );
    fn broadcast_shape(self, other: (usize, )) -> Option<Self::Output> {
        Some(((self.0 == other.0).then_some(self.0)?, ))
    }
}


impl<const N1: usize, > BroadcastShape<(Const<N1>, NewAxis, )> for (usize, )
{
    type Output = (Const<N1>, usize, );
    fn broadcast_shape(self, _other: (Const<N1>, NewAxis, )) -> Option<Self::Output> {
        Some((Const, self.0, ))
    }
}


impl BroadcastShape<(usize, NewAxis, )> for (usize, )
{
    type Output = (usize, usize, );
    fn broadcast_shape(self, other: (usize, NewAxis, )) -> Option<Self::Output> {
        Some((other.0, self.0, ))
    }
}


impl BroadcastShape<()> for (usize, )
{
    type Output = (usize, );
    fn broadcast_shape(self, _other: ()) -> Option<Self::Output> {
        Some((self.0, ))
    }
}


impl<const N0: usize, const N1: usize, > IntoIndex<(Const<N1>, Const<N0>, )> for (Const<N1>, NewAxis, )
{
    fn into_index(index: [usize; 1]) -> [usize; 2] {
        [0, index[0], ]
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N1>, Const<N0>, )> for (Const<N1>, NewAxis, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N1>, Const<N0>, )) -> Option<Self::Output> {
        Some((Const, Const, ))
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(usize, Const<N0>, )> for (Const<N1>, NewAxis, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, other: (usize, Const<N0>, )) -> Option<Self::Output> {
        Some(((N1 == other.0).then_some(Const)?, Const, ))
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N0>, )> for (Const<N1>, NewAxis, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N0>, )) -> Option<Self::Output> {
        Some((Const, Const, ))
    }
}


impl<const N1: usize, > IntoIndex<(Const<N1>, usize, )> for (Const<N1>, NewAxis, )
{
    fn into_index(index: [usize; 1]) -> [usize; 2] {
        [0, index[0], ]
    }
}


impl<const N1: usize, > BroadcastShape<(Const<N1>, usize, )> for (Const<N1>, NewAxis, )
{
    type Output = (Const<N1>, usize, );
    fn broadcast_shape(self, other: (Const<N1>, usize, )) -> Option<Self::Output> {
        Some((Const, other.1, ))
    }
}


impl<const N1: usize, > BroadcastShape<(usize, usize, )> for (Const<N1>, NewAxis, )
{
    type Output = (Const<N1>, usize, );
    fn broadcast_shape(self, other: (usize, usize, )) -> Option<Self::Output> {
        Some(((N1 == other.0).then_some(Const)?, other.1, ))
    }
}


impl<const N1: usize, > BroadcastShape<(usize, )> for (Const<N1>, NewAxis, )
{
    type Output = (Const<N1>, usize, );
    fn broadcast_shape(self, other: (usize, )) -> Option<Self::Output> {
        Some((Const, other.0, ))
    }
}


impl<const N1: usize, > ShapeEq<(Const<N1>, NewAxis, )> for (Const<N1>, NewAxis, )
{
    fn shape_eq(&self, _other: &(Const<N1>, NewAxis, )) -> bool {
        true
    }
}


impl<const N1: usize, > IntoIndex<(Const<N1>, NewAxis, )> for (Const<N1>, NewAxis, )
{
    fn into_index(index: [usize; 1]) -> [usize; 1] {
        [index[0], ]
    }
}


impl<const N1: usize, > BroadcastShape<(Const<N1>, NewAxis, )> for (Const<N1>, NewAxis, )
{
    type Output = (Const<N1>, NewAxis, );
    fn broadcast_shape(self, _other: (Const<N1>, NewAxis, )) -> Option<Self::Output> {
        Some((Const, NewAxis, ))
    }
}


impl<const N1: usize, > ShapeEq<(usize, NewAxis, )> for (Const<N1>, NewAxis, )
{
    fn shape_eq(&self, other: &(usize, NewAxis, )) -> bool {
        N1 == other.0
    }
}


impl<const N1: usize, > BroadcastShape<(usize, NewAxis, )> for (Const<N1>, NewAxis, )
{
    type Output = (Const<N1>, NewAxis, );
    fn broadcast_shape(self, other: (usize, NewAxis, )) -> Option<Self::Output> {
        Some(((N1 == other.0).then_some(Const)?, NewAxis, ))
    }
}


impl<const N1: usize, > BroadcastShape<()> for (Const<N1>, NewAxis, )
{
    type Output = (Const<N1>, NewAxis, );
    fn broadcast_shape(self, _other: ()) -> Option<Self::Output> {
        Some((Const, NewAxis, ))
    }
}


impl<const N0: usize, const N1: usize, > IntoIndex<(Const<N1>, Const<N0>, )> for (usize, NewAxis, )
{
    fn into_index(index: [usize; 1]) -> [usize; 2] {
        [0, index[0], ]
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N1>, Const<N0>, )> for (usize, NewAxis, )
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N1>, Const<N0>, )) -> Option<Self::Output> {
        Some(((self.0 == N1).then_some(Const)?, Const, ))
    }
}


impl<const N0: usize, > IntoIndex<(usize, Const<N0>, )> for (usize, NewAxis, )
{
    fn into_index(index: [usize; 1]) -> [usize; 2] {
        [0, index[0], ]
    }
}


impl<const N0: usize, > BroadcastShape<(usize, Const<N0>, )> for (usize, NewAxis, )
{
    type Output = (usize, Const<N0>, );
    fn broadcast_shape(self, other: (usize, Const<N0>, )) -> Option<Self::Output> {
        Some(((self.0 == other.0).then_some(self.0)?, Const, ))
    }
}


impl<const N0: usize, > BroadcastShape<(Const<N0>, )> for (usize, NewAxis, )
{
    type Output = (usize, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N0>, )) -> Option<Self::Output> {
        Some((self.0, Const, ))
    }
}


impl<const N1: usize, > IntoIndex<(Const<N1>, usize, )> for (usize, NewAxis, )
{
    fn into_index(index: [usize; 1]) -> [usize; 2] {
        [0, index[0], ]
    }
}


impl<const N1: usize, > BroadcastShape<(Const<N1>, usize, )> for (usize, NewAxis, )
{
    type Output = (Const<N1>, usize, );
    fn broadcast_shape(self, other: (Const<N1>, usize, )) -> Option<Self::Output> {
        Some(((self.0 == N1).then_some(Const)?, other.1, ))
    }
}


impl IntoIndex<(usize, usize, )> for (usize, NewAxis, )
{
    fn into_index(index: [usize; 1]) -> [usize; 2] {
        [0, index[0], ]
    }
}


impl BroadcastShape<(usize, usize, )> for (usize, NewAxis, )
{
    type Output = (usize, usize, );
    fn broadcast_shape(self, other: (usize, usize, )) -> Option<Self::Output> {
        Some(((self.0 == other.0).then_some(self.0)?, other.1, ))
    }
}


impl BroadcastShape<(usize, )> for (usize, NewAxis, )
{
    type Output = (usize, usize, );
    fn broadcast_shape(self, other: (usize, )) -> Option<Self::Output> {
        Some((self.0, other.0, ))
    }
}


impl<const N1: usize, > ShapeEq<(Const<N1>, NewAxis, )> for (usize, NewAxis, )
{
    fn shape_eq(&self, _other: &(Const<N1>, NewAxis, )) -> bool {
        self.0 == N1
    }
}


impl<const N1: usize, > IntoIndex<(Const<N1>, NewAxis, )> for (usize, NewAxis, )
{
    fn into_index(index: [usize; 1]) -> [usize; 1] {
        [index[0], ]
    }
}


impl<const N1: usize, > BroadcastShape<(Const<N1>, NewAxis, )> for (usize, NewAxis, )
{
    type Output = (Const<N1>, NewAxis, );
    fn broadcast_shape(self, _other: (Const<N1>, NewAxis, )) -> Option<Self::Output> {
        Some(((self.0 == N1).then_some(Const)?, NewAxis, ))
    }
}


impl ShapeEq<(usize, NewAxis, )> for (usize, NewAxis, )
{
    fn shape_eq(&self, other: &(usize, NewAxis, )) -> bool {
        self.0 == other.0
    }
}


impl IntoIndex<(usize, NewAxis, )> for (usize, NewAxis, )
{
    fn into_index(index: [usize; 1]) -> [usize; 1] {
        [index[0], ]
    }
}


impl BroadcastShape<(usize, NewAxis, )> for (usize, NewAxis, )
{
    type Output = (usize, NewAxis, );
    fn broadcast_shape(self, other: (usize, NewAxis, )) -> Option<Self::Output> {
        Some(((self.0 == other.0).then_some(self.0)?, NewAxis, ))
    }
}


impl BroadcastShape<()> for (usize, NewAxis, )
{
    type Output = (usize, NewAxis, );
    fn broadcast_shape(self, _other: ()) -> Option<Self::Output> {
        Some((self.0, NewAxis, ))
    }
}


impl<const N0: usize, const N1: usize, > IntoIndex<(Const<N1>, Const<N0>, )> for ()
{
    fn into_index(_index: [usize; 0]) -> [usize; 2] {
        [0, 0, ]
    }
}


impl<const N0: usize, const N1: usize, > BroadcastShape<(Const<N1>, Const<N0>, )> for ()
{
    type Output = (Const<N1>, Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N1>, Const<N0>, )) -> Option<Self::Output> {
        Some((Const, Const, ))
    }
}


impl<const N0: usize, > IntoIndex<(usize, Const<N0>, )> for ()
{
    fn into_index(_index: [usize; 0]) -> [usize; 2] {
        [0, 0, ]
    }
}


impl<const N0: usize, > BroadcastShape<(usize, Const<N0>, )> for ()
{
    type Output = (usize, Const<N0>, );
    fn broadcast_shape(self, other: (usize, Const<N0>, )) -> Option<Self::Output> {
        Some((other.0, Const, ))
    }
}


impl<const N0: usize, > IntoIndex<(Const<N0>, )> for ()
{
    fn into_index(_index: [usize; 0]) -> [usize; 1] {
        [0, ]
    }
}


impl<const N0: usize, > BroadcastShape<(Const<N0>, )> for ()
{
    type Output = (Const<N0>, );
    fn broadcast_shape(self, _other: (Const<N0>, )) -> Option<Self::Output> {
        Some((Const, ))
    }
}


impl<const N1: usize, > IntoIndex<(Const<N1>, usize, )> for ()
{
    fn into_index(_index: [usize; 0]) -> [usize; 2] {
        [0, 0, ]
    }
}


impl<const N1: usize, > BroadcastShape<(Const<N1>, usize, )> for ()
{
    type Output = (Const<N1>, usize, );
    fn broadcast_shape(self, other: (Const<N1>, usize, )) -> Option<Self::Output> {
        Some((Const, other.1, ))
    }
}


impl IntoIndex<(usize, usize, )> for ()
{
    fn into_index(_index: [usize; 0]) -> [usize; 2] {
        [0, 0, ]
    }
}


impl BroadcastShape<(usize, usize, )> for ()
{
    type Output = (usize, usize, );
    fn broadcast_shape(self, other: (usize, usize, )) -> Option<Self::Output> {
        Some((other.0, other.1, ))
    }
}


impl IntoIndex<(usize, )> for ()
{
    fn into_index(_index: [usize; 0]) -> [usize; 1] {
        [0, ]
    }
}


impl BroadcastShape<(usize, )> for ()
{
    type Output = (usize, );
    fn broadcast_shape(self, other: (usize, )) -> Option<Self::Output> {
        Some((other.0, ))
    }
}


impl<const N1: usize, > IntoIndex<(Const<N1>, NewAxis, )> for ()
{
    fn into_index(_index: [usize; 0]) -> [usize; 1] {
        [0, ]
    }
}


impl<const N1: usize, > BroadcastShape<(Const<N1>, NewAxis, )> for ()
{
    type Output = (Const<N1>, NewAxis, );
    fn broadcast_shape(self, _other: (Const<N1>, NewAxis, )) -> Option<Self::Output> {
        Some((Const, NewAxis, ))
    }
}


impl IntoIndex<(usize, NewAxis, )> for ()
{
    fn into_index(_index: [usize; 0]) -> [usize; 1] {
        [0, ]
    }
}


impl BroadcastShape<(usize, NewAxis, )> for ()
{
    type Output = (usize, NewAxis, );
    fn broadcast_shape(self, other: (usize, NewAxis, )) -> Option<Self::Output> {
        Some((other.0, NewAxis, ))
    }
}


impl ShapeEq<()> for ()
{
    fn shape_eq(&self, _other: &()) -> bool {
        true
    }
}


impl IntoIndex<()> for ()
{
    fn into_index(_index: [usize; 0]) -> [usize; 0] {
        []
    }
}


impl BroadcastShape<()> for ()
{
    type Output = ();
    fn broadcast_shape(self, _other: ()) -> Option<Self::Output> {
        Some(())
    }
}

