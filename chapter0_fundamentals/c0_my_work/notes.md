# Catch Up

bad formatting bc theyre my notes


## broadcasting
broadcasting lets you combine arrays of different shapes

simplest case: scalars
([1,2,3]) + 10 stretches 10 to [10,10,10] and outputs [11,12,13]

to combine two arrays, numpy lines up their shapes from the right and checks dimensions. theyre compatible of equal or if one of them is 1, 1 gets stretched to match the other. if they have fewer dimensions, pad the left dimensions with 1s

ex: matrix + vector
m + v
m: (2,3), v (3)
v -> (1,3) m + v -> (2,3)
so v becomes (a,b,c) -> ([a,b,c],[a,b,c])

ex: column vector + row vector
a = [1,2,3] a: (3,)
b = ([10],[20],[30]) b: (3, 1)
a -> (1,3) b -> (3,1) so a+b = (3,3)
becomes ([1,2,3],[1,2,3],[1,2,3]) + ([10,10,10],[20,20,20],[30,30,30]) = ([11,12,13],[21,22,23],[31,32,33])


## tensors
