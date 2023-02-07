############################################################################################
# Constants
"RNG for sampling based solutions"
const _RNG = Random.Xoshiro(123)  # shorthand
Random.seed!(_RNG, 1)

"Tolerance for stochastic solutions"
const _TOL = 1.0e-6

"Number of samples"
N = 10^3

############################################################################################
# Initiate DefaultFlatten structs for strict and flexible (AD) type conversion:
outputtypes = [Float32, Float64]
flattentypes = [FlattenContinuous(), FlattenAll()]
unflattenmethods = [UnflattenStrict(), UnflattenFlexible()]

############################################################################################
#Probabilistic Parameters - Some selected distributions
struct ProbModel <: ModelName end
val_dist = (
    ## Normal distribution
    d_a1=Param(Distributions.Normal(), 2.0),
    d_a2=Param(Distributions.Normal{Float32}(Float32(1.0), Float32(2.0)), Float32(2.0), ),
    d_a3=Param([Distributions.Normal(), Distributions.Normal(2.0, 3.0), ], [2.0, 2.0]),
    d_a5=Param(
        [
            Distributions.Normal{Float16}(Float16(1.0), Float16(2.0)),
            Distributions.Normal{Float16}(Float16(1.0), Float16(2.0)),
        ],
        Float16.([52.0, 53.0]),
    ),
    d_a6=Param(

        [
            [Distributions.Normal(), Distributions.Normal(2.0, 3.0)],
            [Distributions.Normal(), Distributions.Normal(2.0, 3.0)],
        ],
        [[2.0, 2.0], [2.0, 2.0]],
    ),
    d_a7=Param(Distributions.truncated(Distributions.Normal(2.0, 3.0), 0.0, 10.0), 5.0, ),
    d_a8=Param(
        [
            Distributions.truncated(Distributions.Normal(2.0, 3.0), 0.0, 10.0),
            Distributions.truncated(Distributions.Normal(2.0, 3.0), 0.0, 10.0),
        ],
        [2.0, 2.0],
    ),
    ## Multivariate Normal
    d_b1=Param(
        Distributions.MvNormal([2.0, 3.0], Diagonal(map(abs2, [1.0, 1.0]))),
    [2.0, 2.0],
    ),
    d_b2=Param(
        [
            Distributions.MvNormal([2.0, 3.0], Diagonal(map(abs2, [1.0, 1.0]))),
            Distributions.MvNormal(Diagonal(map(abs2, [1.0, 1.0]))),
        ],
        [[2.0, 2.0], [2.0, 2.0]],
    )
    ,
    d_b3=Param(
        [
            Distributions.MvNormal(Float32.([1, 1.0]), Float32.([5.0 0.3; 0.3 5.0])),
            Distributions.MvNormal(Float32.([1, 1.0]), Float32.([5.0 0.3; 0.3 5.0])),
        ],
        [Float32.([54.0, 55.0]), Float32.([56.0, 57.0])],
    ),
    ## Gamma
    d_c1=Param(Distributions.Gamma(2, 3), 2.0, ),
    d_c2=Param([Distributions.Gamma(2, 3), Distributions.Gamma(2, 3)], [2.0, 2.0], ),
    ## Exponential
    d_d1=Param(Distributions.Exponential(2), 2.0, ),
    d_d2=Param([Distributions.Exponential(2), Distributions.Exponential(2)], [2.0, 2.0], ),
    ## Dirichlet Distribution
    d_e1=Param(Distributions.Dirichlet(2, 2.0), [0.1, 0.9], ),
    d_e2=Param(
        [Distributions.Dirichlet(2, 2.0), Distributions.Dirichlet(2, 2.0)],
        [[0.1, 0.9], [0.8, 0.2]],
    ),
    d_e3=Param(Distributions.Dirichlet(3, 3.0), [0.2, 0.3, 0.5], ),
    d_e4=Param(
        [Distributions.Dirichlet(3, 3.0), Distributions.Dirichlet(3, 3.0)],
        [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]],
    ),
    ## Categorical
    d_f1=Param(Distributions.Categorical(100), 5, ),
    #f2 = Param(5, Distributions.Categorical( Float32.( repeat([1/100], 100) ) ),
    d_f3=Param([Distributions.Categorical(100), Distributions.Categorical(100)], [10, 10], ),
    d_f4=Param(
        [
            [Distributions.Categorical(40), Distributions.Categorical(40)],
            [Distributions.Categorical(40), Distributions.Categorical(40)],
        ],
        [[34, 34], [34, 34]],
    ),
    ## Poisson
    d_g1=Param(Distributions.Poisson(100), 5, ),
    d_g2=Param([Distributions.Poisson(100), Distributions.Poisson(100)], [10, 10], ),
    ## Negative Binomial ~ Does not work for Zygote (foreign function call)
    #        d_h1 = Param(5, Distributions.NegativeBinomial(.5, .5)),
    #        d_h2 = Param([10, 10], [Distributions.NegativeBinomial(.5, .5), Distributions.NegativeBinomial(.5, .5)]),
    ## Uniform
    d_i1=Param(Distributions.Uniform(-5.0, 5.0), 1.0, ),
    #~ Not correctly sampled as Float16 from Distributions package.
    #d_i2 = Param(Float16(5.), Distributions.Uniform(Float16(-10), Float16(10))),
    d_i3=Param(
        [Distributions.Uniform(-5.0, 5.0), Distributions.Uniform(-5.0, 5.0)],
        [3.0, 3.0],
    ),
    ## InverseWishart ~ Does not work with ReverseDiff due to Diag{} type conversion from Bijector
    #        d_j1 = Param([15. .16 ; .16 18.], Distributions.InverseWishart(10., [1. 0. ; 0. 1.])),
    #        d_j2 = Param([[19. .20 ; .20 22.], [23. .24 ; .24 26.]],[Distributions.InverseWishart(10., [1. 0. ; 0. 1.]), Distributions.InverseWishart(10., [1. 0. ; 0. 1.])]),
    #        d_j3 = Param([15. .16 .16 ; .16 18. .16 ; .16 .16 20.], Distributions.InverseWishart(10., [1. 0. .0 ; 0 1 0 ; 0 0. 1.])),
    #        d_j4 = Param([[15. .16 .16 ; .16 18. .16 ; .16 .16 20.], [15. .16 .16 ; .16 18. .16 ; .16 .16 20.]],[Distributions.InverseWishart(10., [1. 0. .0 ; 0 1 0 ; 0 0. 1.]), Distributions.InverseWishart(10., [1. 0. .0 ; 0 1 0 ; 0 0. 1.])]),
    ## Cauchy
    d_k1=Param(Distributions.Cauchy(5.0, 10.0), 5.0, ),
    d_k2=Param(
        [Distributions.Cauchy(5.0, 10.0), Distributions.Cauchy(5.0, 10.0)],
        [2.0, 2.0],
    ),
    ## LKJ
    d_l1=Param(Distributions.LKJ(2, 1.0), [1.0 0.16; 0.16 1.0], ),
    d_l2=Param(
        [Distributions.LKJ(2, 1.0), Distributions.LKJ(2, 1.0)],
        [[1.0 0.20; 0.20 1.0], [1.0 0.24; 0.24 1.0]],
    ),
    d_l3=Param(Distributions.LKJ(3, 1.0), [1.0 0.16 0.16; 0.16 1.0 0.16; 0.16 0.16 1.0], ),
    d_l4=Param(
        [Distributions.LKJ(3, 1.0), Distributions.LKJ(3, 1.0)],
        [
            [1.0 0.16 0.16; 0.16 1.0 0.16; 0.16 0.16 1.0],
            [1.0 0.16 0.16; 0.16 1.0 0.16; 0.16 0.16 1.0],
        ],
    ),
    ## Truncated
    d_m1=Param(Distributions.truncated(Distributions.Normal(5.0, 10.0), 0.0, 20.0), 10.0, ),
    d_m2=Param(Distributions.truncated(Distributions.Cauchy(5.0, 10.0), 0.0, 20.0), 10.0, ),
    ## Product Distributions -> Check if ReverseDiff works with this structure
    d_n1=Param(
        [
            Distributions.product_distribution([Uniform(-1, 1) for _ in Base.OneTo(3)]) for
            _ in Base.OneTo(4)
        ],
        [[i / 10 for i in Base.OneTo(3)] for _ in Base.OneTo(4)],
    ),
)
#val_dist_length = 58

############################################################################################
# Nested Probabilistic Parameter/Models - Some selected distributions
val_dist_nested = (;
    nd_a1=(a=val_dist.d_a1, b=val_dist.d_f1, c=val_dist.d_i3),
    nd_a2=(
        a=(b=val_dist.d_a1, c=val_dist.d_f1, d=val_dist.d_i3),
        e=val_dist.d_a2,
        f=val_dist.d_e2,
    ),
    nd_a3=(
        a=(b=val_dist.d_a1, c=(g=val_dist.d_f1,), d=val_dist.d_i3),
        e=val_dist.d_a2,
        f=val_dist.d_e2,
    ),
)

############################################################################################
# Non-Probabilistic Parameter (Experimental!):
struct ConstrainedModel <: ModelName end
val_constrained = (
    ## Fixed
    # Floats
    c_a1=Param(Fixed(), 1.0, ),
    c_a2=Param(Fixed(), [2.0, 2.0], ),
    c_a3=Param(Fixed(), [14.0 0.14; 0.14 14.0], ),
    c_b1=Param(Fixed(), Float32(26.0), ),
    c_b2=Param(Fixed(), Float16.([28.0, 28.0]), ),
    # Integer - Should not be flattened
    c_c1=Param(Fixed(), 34, ),
    c_c2=Param(Fixed(), [34, 34], ),
    # Fixed buffer parameter
    c_d1=Param(Fixed(), rand(1:100, 100), ),
    c_d2=Param(Fixed(), randn(100), ),
    c_d3=Param(Fixed(), rand(1:100, 10, 10), ),
    c_d4=Param(Fixed(), randn(10, 10), ),
    # Arbitrary AbstractArrays
    c_e1=Param(Fixed(), [[1.0, 2.0], zeros(3, 2, 1)], ),
    c_e2=Param(Fixed(), [[1, 2], [1, 2]], ),
    c_e3=Param(Fixed(), [[1.0, 2.0], [1.0, 2.0]], ),
    c_e4=Param(Fixed(), [(rand(), rand()) for _ in 1:10], ),
    c_e5=Param(Fixed(), [(rand(1:10), rand(1:10)) for _ in 1:10]),
    ## Unconstrained Parameter
    # Floats
    u_a1=Param(Unconstrained(), 1.0, ),
    u_a2=Param(Unconstrained(), [2.0, 2.0], ),
    u_a3=Param(Unconstrained(), [14.0 0.14; 0.14 14.0], ),
    u_b1=Param(Unconstrained(), Float32(26.0), ),
    u_b2=Param(Unconstrained(), Float16.([28.0, 28.0]), ),
    # Fixed buffer parameter
    u_d1=Param(Unconstrained(), rand(1:100, 100), ),
    u_d2=Param(Unconstrained(), randn(100), ),
    u_d3=Param(Unconstrained(), rand(1:100, 10, 10), ),
    u_d4=Param(Unconstrained(), randn(10, 10), ),
    # Arbitrary AbstractArrays
    u_e1=Param(Unconstrained(), [[1.0, 2.0], zeros(3, 2, 1)], ),
    u_e2=Param(Unconstrained(), [[1, 2], [1, 2]], ),
    u_e3=Param(Unconstrained(), [[1.0, 2.0], [1.0, 2.0]], ),
#    u_e4=Param(Unconstrained(), [(rand(), rand()) for _ in 1:10], ),
#    u_e5=Param(Unconstrained(), [(rand(1:10), rand(1:10)) for _ in 1:10], ),
    ## Constrained Parameter -> Scalar only
    con_a1=Param(Constrained(0.0, 2.0), 1.0, ),
    con_b1=Param(Constrained(Float32(20.0), Float32(30.0)), Float32(26.0), ),
)

################################################################################
# Parameter for an example Model
_σ = Distributions.truncated(Distributions.Cauchy(10.0, 10), 0.0, 30.0)
_iwish2 = Distributions.InverseWishart(10.0, [1.0 0.0; 0.0 1.0])
_iwish3 = Distributions.InverseWishart(10.0, [1.0 0.0 0.0; 0 1 0; 0 0.0 1.0])
_ρ = [1.0 0.25; 0.25 1.0]
struct ExampleModel <: ModelName end
_val_examplemodel = (
    μ1=Param(Distributions.Normal(), 1.0, ),
    μ2=Param(Distributions.MvNormal(Diagonal(map(abs2, [1.0, 1.0]))), [2.0, 3.0], ),
    μ3=Param([Distributions.Normal(), Distributions.Normal()], [4.0, 5.0], ),
    μ4=Param(
        [
            Distributions.MvNormal([2.0, 3.0], Diagonal(map(abs2, [1.0, 1.0]))),
            Distributions.MvNormal([1.0, 2.0], [3.0 0.4; 0.4 5.0]),
        ],
        [[6.0, 7.0], [8.0, 9.0]],
    ),
    σ1=Param(Distributions.Gamma(), 10.0, ),
    σ3=Param([Distributions.Gamma(), Distributions.Gamma()], [5.0, 5.0], ),
    σ2=Param([_σ, _σ], [5.0, 5.0], ),
    ρ2=Param(Distributions.LKJ(2, 1.0), copy(_ρ), ),
    σ4=Param([[_σ, _σ], [_σ, _σ]], [[5.0, 5.0], [5.0, 5.0]]),
    ρ4=Param([Distributions.LKJ(2, 1.0), Distributions.LKJ(2, 1.0)], [copy(_ρ), copy(_ρ)], ),
    p=Param(Distributions.Dirichlet(3, 3.0), [0.2, 0.3, 0.5],),
)
#_val_examplemodel_length = 23

################################################################################
# Test Gradients of Model with custom function

# Parameter that may work in lower dimensions
struct LowerDims <: ModelName end
_val_lowerdims = (;
    ## Dirichlet
    ldim_e1=Param(Distributions.Dirichlet(2, 2.0), [0.1, 0.9], ),
    ldim_e2=Param(
        [Distributions.Dirichlet(2, 2.0), Distributions.Dirichlet(2, 2.0)],
        [[0.1, 0.9], [0.8, 0.2]],
    ),
    ldim_e3=Param(Distributions.Dirichlet(3, 3.0), [0.2, 0.3, 0.5], ),
    ldim_e4=Param(
        [Distributions.Dirichlet(3, 3.0), Distributions.Dirichlet(3, 3.0)],
        [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]],
    ),
    ## LKJ
    ldim_l1=Param(Distributions.LKJ(2, 1.0), [1.0 0.16; 0.16 1.0], ),
    ldim_l2=Param(
        [Distributions.LKJ(2, 1.0), Distributions.LKJ(2, 1.0)],
        [[1.0 0.20; 0.20 1.0], [1.0 0.24; 0.24 1.0]],
    ),
    ldim_l3=Param(
        Distributions.LKJ(3, 1.0),
        [1.0 0.16 0.16; 0.16 1.0 0.16; 0.16 0.16 1.0],
        ),
    ldim_l4=Param(
        [Distributions.LKJ(3, 1.0), Distributions.LKJ(3, 1.0)],
        [
            [1.0 0.16 0.16; 0.16 1.0 0.16; 0.16 0.16 1.0],
            [1.0 0.16 0.16; 0.16 1.0 0.16; 0.16 0.16 1.0],
        ],
    ),
    ## InverseWishart
    #        ldim_j1 = Param([15. .16 ; .16 18.], _iwish2),
    #        ldim_j2 = Param([[19. .20 ; .20 22.], [23. .24 ; .24 26.]],[_iwish2, _iwish2]),
    #        ldim_j3 = Param([15. .16 .16 ; .16 18. .16 ; .16 .16 20.], _iwish3),
    #        ldim_j4 = Param([[15. .16 .16 ; .16 18. .16 ; .16 .16 20.], [15. .16 .16 ; .16 18. .16 ; .16 .16 20.]],[_iwish3, _iwish3]),
)
