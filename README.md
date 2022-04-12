# ModelWrappers

[![Documentation, Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://paschermayr.github.io/ModelWrappers.jl/)
[![Build Status](https://github.com/paschermayr/ModelWrappers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/paschermayr/ModelWrappers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/paschermayr/ModelWrappers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/paschermayr/ModelWrappers.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

ModelWrappers.jl is a utility package that makes it easier to work with Model parameters stated as (nested) NamedTuples. It handles
1. flattening/unflattening model parameter fields of arbitrary dimensions.
2. constraining/unconstraining parameter (if a corresponding constraint is provided).
3. using Automatic Differentation for Model parameter given as a `NamedTuple` for a user specified function, i.e., a log-posterior distribution.

<!---

[BaytesMCMC.jl](xxx)
[BaytesFilters.jl](xxx)
[BaytesPMCMC.jl](xxx)
[BaytesSMC.jl](xxx)
[Baytes.jl](xxx)
-->

## Flattening/Unflattening Model Parameter

ModelWrappers.jl allows you to `flatten` a (nested) NamedTuple to a vector, and also returns an `unflatten` function to convert a vector back to a NamedTuple. By default, discrete parameter are not flattened and the default flatten type is `Float64`. One can construct flatten/unflatten via a `ReConstructor`.
```julia
using ModelWrappers
myparameter = (a = Float32(1.), b = 2, c = [3., 4.], d = [5, 6])
reconstruct = ReConstructor(myparameter)
vals_vec = flatten(reconstruct, myparameter) #Vector{Float64} with 3 elements (1., 3., 4.)
vals = unflatten(reconstruct, vals_vec) #(a = 1.0f0, b = 2, c = [3.0, 4.0], d = [5, 6])
```

You can adjust these settings by using the `FlattenDefault` struct. For instance, the following settings will map `myparameter` to a `Float16` vector and also flatten the Integer values.
```julia
flattendefault = FlattenDefault(; output = Float16, flattentype = FlattenAll())
reconstruct = ReConstructor(flattendefault, myparameter)
vals_vec = flatten(reconstruct, myparameter) #Vector{Float16} with 6 elements (1., 2., 3., 4., 5., 6.)
vals = unflatten(reconstruct, vals_vec) #(a = 1.0f0, b = 2, c = [3.0, 4.0], d = [5, 6])
```

Flatten/Unflatten can also be used for Automatic Differentiation. The functions `flattenAD` and `unflattenAD` return output based on the input type. Check the differences to the first two cases in this example:
```julia
myparameter = (a = Float32(1.), b = 2, c = [3., 4.], d = [5, 6])
flattendefault = FlattenDefault(; output = Float32)
reconstruct = ReConstructor(flattendefault, myparameter)
vals_vec = flattenAD(reconstruct, myparameter) #Vector{Float64} with 3 elements (1., 3., 4.)
vals = unflattenAD(reconstruct, vals_vec) #(a = 1.0, b = 2.0, c = [3.0, 4.0], d = [5.0, 6.0])
```

A `ReConstructor` will assign buffers for `flatten` and `unflatten`, so most operations can be performed without allocations. Unflatten can usually be performed free of most allocations, even if arrays are involved:
```julia
using BenchmarkTools
myparameter2 = (a = Float32(1.), b = 2, c = [3., 4.], d = [5, 6], e = randn(1000), f = rand(1:2, 1000), g = randn(1000, 2))
reconstruct = ReConstructor(myparameter2)
vals_vec = flatten(reconstruct, myparameter2)
vals_vec #Vector{Float64} with 3003 element
@btime unflatten($reconstruct, $vals_vec)   # 419.095 ns (0 allocations: 0 bytes)
@btime flatten($reconstruct, $myparameter2) # 3.475 μs (8 allocations: 39.83 KiB)
```

Note that it is possible to nest NamedTuples, and use arbitrary Array-of-Arrays structures for your parameter, but this will often come with a performance penalty:
```julia
myparameter3 = (a = myparameter, b = (c = (d = myparameter2, ), ), e = [rand(10), rand(15), rand(20)])
reconstruct = ReConstructor(myparameter3)
vals_vec = flatten(reconstruct, myparameter3)
vals_vec #Vector{Float64} with 3051 element
@btime unflatten($reconstruct, $vals_vec)   # 1.220 μs (32 allocations: 3.19 KiB)
@btime flatten($reconstruct, $myparameter3) # 7.275 μs (19 allocations: 88.17 KiB)
```

## Constraining/Unconstraining Model Parameter

Consider now the following problem: you have a model that consists of various (unknown) parameters and you want to estimate these parameter with a custom algorithm. Many common algorithms not only require you to take a vector as function argument, but also require you to know in which space your parameter operate.

If a corresponding prior distribution is provided, ModelWrappers.jl allows you to efficiently constrain and unconstrain your parameter tuple. To do so, one can initiate a `Param` struct, which is a temporary constructor that checks if the package can handle the (value, constraint) combination. The initial `NamedTuple` can then be wrapped in a `ModelWrapper` struct.
```julia
using Distributions
myparameter4 = (μ = Param(0.0, Normal()), σ = Param(1.0, Gamma()))
mymodel = ModelWrapper(myparameter4)
```

Note that providing a prior distribution in `Param` will just assign a bijector to the parameter, so instead of providing a prior distribution, one may provide a Bijector directly. The code below constructs the same model as above:
```julia
using Bijectors
myparameter_bij = (μ = Param(0.0, Identity{0}()), σ = Param(1.0, Bijectors.Log{0}()))
mymodel_bij = ModelWrapper(myparameter_bij)
```

Valid constraint options for a `Param` struct at the moment include
1. a bijector from [Bijectors.jl](https://github.com/TuringLang/Bijectors.jl),
2. all distributions that work with [Bijectors.jl](https://github.com/TuringLang/Bijectors.jl),
3. a `Fixed` struct, which keeps `val` fixed and excludes it from flatten/unflatten,
4. an `Unconstrained` struct, which flattens `val` without taking into account any constraint,
5. and a `Constrained` struct, which flattens `val` without taking into account any constraint, but will take into account the constraints when constraining values.
```julia
myparameter_constraints = (
    μ = Param(0.0, Normal()),
    σ = Param(1.0, Bijectors.Log{0}()),
    buffer1 = Param(zeros(Int64, 2,3,4), Fixed()),
    buffer2 = Param([zeros(10), zeros(20)], Unconstrained()),
    buffer3 = Param(3., Constrained(1., 5.))
)
model_constraints = ModelWrapper(myparameter_constraints)
flatten(model_constraints) #Vector{Float64} with 33 elements
```

A `ModelWrapper` struct is mutable, and contains the values of your `NamedTuple` field. Values can be flattened or unconstrained, and may be updated by new values/samples. Also, when a `ModelWrapper` struct is created, an unflatten function for strict and variable type conversion is stored. To show this, we will a create `ModelWrapper` struct, flatten its values, and update the struct with new values:

```julia
using Distributions, Random
_rng = MersenneTwister(2)
myparameter4 = (μ = Param(0.0, Normal()), σ = Param(1.0, Gamma()))
mymodel = ModelWrapper(myparameter4)
#Flatten/Unconstrain Model parameter
vals_vec = flatten(mymodel) #Vector{Float64} with 2 elements
unconstrain(mymodel) #(μ = 0.0, σ = 0.0)
unconstrain_flatten(mymodel) #Vector{Float64} with 2 elements

#Unflatten/Constrain proposed parameter from unconstrained space
θ_proposed = randn(_rng, length(vals_vec)) #Vector{Float64} with 2 elements
ModelWrappers.unflatten(mymodel, θ_proposed) #(μ = 0.7396206598864331, σ = -0.7445071021408705)
unflatten_constrain(mymodel, θ_proposed) #(μ = 0.7396206598864331, σ = 0.4749683531374296)

#Replacing current model parameter with proposed parameter
mymodel.val #(μ = 0.0, σ = 1.0)
unflatten_constrain!(mymodel, θ_proposed)
mymodel.val #(μ = 0.7396206598864331, σ = 0.4749683531374296)
```

## Using Automatic Differentiation with a `ModelWrapper`

`ModelWrappers.jl` supports the usage of various Automatic Differentiation backends by providing an immutable `Objective` struct that contains your `ModelWrapper`, data, and all parameter that you want to get derivative information from. `Objective` is a functor, and you can manually add a target function wrt your original parameter `NamedTuple` that should be included in the AD call, i.e., a log-posterior density.

Let us work with the model from before. We first sample data, create the objective and then define a function that we want to use AD for:
```julia
using UnPack
#Create Model and data
myparameter4 = (μ = Param(0.0, Normal()), σ = Param(1.0, Gamma()))
mymodel = ModelWrapper(myparameter4)
data = randn(1000)

#Create objective for both μ and σ and define a target function for it
myobjective = Objective(mymodel, data, (:μ, :σ))
function (objective::Objective{<:ModelWrapper{BaseModel}})(θ::NamedTuple)
	@unpack data = objective
	lprior = Distributions.logpdf(Distributions.Normal(),θ.μ) + Distributions.logpdf(Distributions.Exponential(), θ.σ)
    llik = sum(Distributions.logpdf( Distributions.Normal(θ.μ, θ.σ), data[iter] ) for iter in eachindex(data))
	return lprior + llik
end
```

`myobjective` can take a vector from an unconstrained space as input, constrains and converts the argument to a `NamedTuple`, checks if all conversions are finite, and adds all eventual Jacobian adjustments from the transformations before your target function is evaluated. This can usually be done efficiently:
```julia
#Sample new parameter, and evaluate target function wrt to Vector (not NamedTuple)
θ_proposed = randn(_rng, length(vals_vec))
myobjective(θ_proposed)

#Functor call wrt NamedTuple parameter
@btime $myobjective($mymodel.val) #6.420 μs (0 allocations: 0 bytes)
#Functor call wrt proposed Parameter Vector
@btime $myobjective($θ_proposed) #6.480 μs (0 allocations: 0 bytes)
```

`Objective` can also be called from various AD frameworks:
```julia
using ForwardDiff, ReverseDiff, Zygote
grad_fwd = ForwardDiff.gradient(myobjective, θ_proposed)
grad_rvd = ReverseDiff.gradient(myobjective, θ_proposed)
grad_zyg = Zygote.gradient(myobjective, θ_proposed)
all(grad_fwd .≈ grad_rvd .≈ grad_zyg[1]) #true
```
<!---

## Using Soss.jl with ModelWrappers.jl (Experimental)

Instead of manually definining parameter distributions and a target function, ModelWrappers.jl can be used with Soss.jl to obtain all information from a Soss `@model`:

```julia
using Soss
m = @model n begin
    μ ~ Distributions.Normal()
    σ ~ Distributions.Gamma()
    data ~ Distributions.Normal(μ, σ) |> iid(n)
    return (; data)
end
posterior =  m((μ = 0.0, σ = 1.0, n = length(data))) | (data = data,)
model_soss = ModelWrapper(posterior)
objective_soss = Objective(model_soss)

grad_fwd_soss = ForwardDiff.gradient(objective_soss, θ_proposed)
grad_rvd_soss = ReverseDiff.gradient(objective_soss, θ_proposed)
grad_zyg_soss = Zygote.gradient(objective_soss, θ_proposed)
all(grad_fwd_soss .≈ grad_rvd_soss .≈ grad_zyg_soss[1]) #true

objective_soss(mymodel.val) ≈ myobjective(mymodel.val) #true
all(grad_fwd .≈ grad_rvd .≈ grad_zyg[1] .≈ grad_fwd_soss .≈ grad_rvd_soss .≈ grad_zyg_soss[1]) #true
```
-->
## Going Forward

This package is still highly experimental - suggestions and comments are always welcome!

<!---
# Citing Baytes.jl

If you use Baytes.jl for your own research, please consider citing the following publication: ...
-->
