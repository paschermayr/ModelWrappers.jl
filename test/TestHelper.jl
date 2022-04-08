############################################################################################
# Constants
"RNG for sampling based solutions"
const _RNG = Random.MersenneTwister(1)   # shorthand
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
    d_a1=Param(2.0, Distributions.Normal()),
    d_a2=Param(Float32(2.0), Distributions.Normal{Float32}(Float32(1.0), Float32(2.0))),
    d_a3=Param([2.0, 2.0], [Distributions.Normal(), Distributions.Normal(2.0, 3.0)]),
    d_a5=Param(
        Float16.([52.0, 53.0]),
        [
            Distributions.Normal{Float16}(Float16(1.0), Float16(2.0)),
            Distributions.Normal{Float16}(Float16(1.0), Float16(2.0)),
        ],
    ),
    d_a6=Param(
        [[2.0, 2.0], [2.0, 2.0]],
        [
            [Distributions.Normal(), Distributions.Normal(2.0, 3.0)],
            [Distributions.Normal(), Distributions.Normal(2.0, 3.0)],
        ],
    ),
    d_a7=Param(5.0, Distributions.truncated(Distributions.Normal(2.0, 3.0), 0.0, 10.0)),
    d_a8=Param(
        [2.0, 2.0],
        [
            Distributions.truncated(Distributions.Normal(2.0, 3.0), 0.0, 10.0),
            Distributions.truncated(Distributions.Normal(2.0, 3.0), 0.0, 10.0),
        ],
    ),
    ## Multivariate Normal
    d_b1=Param(
        [2.0, 2.0], Distributions.MvNormal([2.0, 3.0], Diagonal(map(abs2, [1.0, 1.0])))
    ),
    d_b2=Param(
        [[2.0, 2.0], [2.0, 2.0]],
        [
            Distributions.MvNormal([2.0, 3.0], Diagonal(map(abs2, [1.0, 1.0]))),
            Distributions.MvNormal(Diagonal(map(abs2, [1.0, 1.0]))),
        ],
    )
    ,
    d_b3=Param(
        [Float32.([54.0, 55.0]), Float32.([56.0, 57.0])],
        [
            Distributions.MvNormal(Float32.([1, 1.0]), Float32.([5.0 0.3; 0.3 5.0])),
            Distributions.MvNormal(Float32.([1, 1.0]), Float32.([5.0 0.3; 0.3 5.0])),
        ],
    ),
    ## Gamma
    d_c1=Param(2.0, Distributions.Gamma(2, 3)),
    d_c2=Param([2.0, 2.0], [Distributions.Gamma(2, 3), Distributions.Gamma(2, 3)]),
    ## Exponential
    d_d1=Param(2.0, Distributions.Exponential(2)),
    d_d2=Param([2.0, 2.0], [Distributions.Exponential(2), Distributions.Exponential(2)]),
    ## Dirichlet Distribution
    d_e1=Param([0.1, 0.9], Distributions.Dirichlet(2, 2.0)),
    d_e2=Param(
        [[0.1, 0.9], [0.8, 0.2]],
        [Distributions.Dirichlet(2, 2.0), Distributions.Dirichlet(2, 2.0)],
    ),
    d_e3=Param([0.2, 0.3, 0.5], Distributions.Dirichlet(3, 3.0)),
    d_e4=Param(
        [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]],
        [Distributions.Dirichlet(3, 3.0), Distributions.Dirichlet(3, 3.0)],
    ),
    ## Categorical
    d_f1=Param(5, Distributions.Categorical(100)),
    #f2 = Param(5, Distributions.Categorical( Float32.( repeat([1/100], 100) ) ),
    d_f3=Param([10, 10], [Distributions.Categorical(100), Distributions.Categorical(100)]),
    d_f4=Param(
        [[34, 34], [34, 34]],
        [
            [Distributions.Categorical(40), Distributions.Categorical(40)],
            [Distributions.Categorical(40), Distributions.Categorical(40)],
        ],
    ),
    ## Poisson
    d_g1=Param(5, Distributions.Poisson(100)),
    d_g2=Param([10, 10], [Distributions.Poisson(100), Distributions.Poisson(100)]),
    ## Negative Binomial ~ Does not work for Zygote (foreign function call)
    #        d_h1 = Param(5, Distributions.NegativeBinomial(.5, .5)),
    #        d_h2 = Param([10, 10], [Distributions.NegativeBinomial(.5, .5), Distributions.NegativeBinomial(.5, .5)]),
    ## Uniform
    d_i1=Param(1.0, Distributions.Uniform(-5.0, 5.0)),
    #~ Not correctly sampled as Float16 from Distributions package.
    #d_i2 = Param(Float16(5.), Distributions.Uniform(Float16(-10), Float16(10))),
    d_i3=Param(
        [3.0, 3.0], [Distributions.Uniform(-5.0, 5.0), Distributions.Uniform(-5.0, 5.0)]
    ),
    ## InverseWishart ~ Does not work with ReverseDiff due to Diag{} type conversion from Bijector
    #        d_j1 = Param([15. .16 ; .16 18.], Distributions.InverseWishart(10., [1. 0. ; 0. 1.])),
    #        d_j2 = Param([[19. .20 ; .20 22.], [23. .24 ; .24 26.]],[Distributions.InverseWishart(10., [1. 0. ; 0. 1.]), Distributions.InverseWishart(10., [1. 0. ; 0. 1.])]),
    #        d_j3 = Param([15. .16 .16 ; .16 18. .16 ; .16 .16 20.], Distributions.InverseWishart(10., [1. 0. .0 ; 0 1 0 ; 0 0. 1.])),
    #        d_j4 = Param([[15. .16 .16 ; .16 18. .16 ; .16 .16 20.], [15. .16 .16 ; .16 18. .16 ; .16 .16 20.]],[Distributions.InverseWishart(10., [1. 0. .0 ; 0 1 0 ; 0 0. 1.]), Distributions.InverseWishart(10., [1. 0. .0 ; 0 1 0 ; 0 0. 1.])]),
    ## Cauchy
    d_k1=Param(5.0, Distributions.Cauchy(5.0, 10.0)),
    d_k2=Param(
        [2.0, 2.0], [Distributions.Cauchy(5.0, 10.0), Distributions.Cauchy(5.0, 10.0)]
    ),
    ## LKJ
    d_l1=Param([1.0 0.16; 0.16 1.0], Distributions.LKJ(2, 1.0)),
    d_l2=Param(
        [[1.0 0.20; 0.20 1.0], [1.0 0.24; 0.24 1.0]],
        [Distributions.LKJ(2, 1.0), Distributions.LKJ(2, 1.0)],
    ),
    d_l3=Param([1.0 0.16 0.16; 0.16 1.0 0.16; 0.16 0.16 1.0], Distributions.LKJ(3, 1.0)),
    d_l4=Param(
        [
            [1.0 0.16 0.16; 0.16 1.0 0.16; 0.16 0.16 1.0],
            [1.0 0.16 0.16; 0.16 1.0 0.16; 0.16 0.16 1.0],
        ],
        [Distributions.LKJ(3, 1.0), Distributions.LKJ(3, 1.0)],
    ),
    ## Truncated
    d_m1=Param(10.0, Distributions.truncated(Distributions.Normal(5.0, 10.0), 0.0, 20.0)),
    d_m2=Param(10.0, Distributions.truncated(Distributions.Cauchy(5.0, 10.0), 0.0, 20.0)),
    ## Product Distributions -> Check if ReverseDiff works with this structure
    d_n1=Param(
        [[i / 10 for i in Base.OneTo(3)] for _ in Base.OneTo(4)],
        [
            Distributions.product_distribution([Uniform(-1, 1) for _ in Base.OneTo(3)]) for
            _ in Base.OneTo(4)
        ],
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
    c_a1=Param(1.0, Fixed()),
    c_a2=Param([2.0, 2.0], Fixed()),
    c_a3=Param([14.0 0.14; 0.14 14.0], Fixed()),
    c_b1=Param(Float32(26.0), Fixed()),
    c_b2=Param(Float16.([28.0, 28.0]), Fixed()),
    # Integer - Should not be flattened
    c_c1=Param(34, Fixed()),
    c_c2=Param([34, 34], Fixed()),
    # Fixed buffer parameter
    c_d1=Param(rand(1:100, 100), Fixed()),
    c_d2=Param(randn(100), Fixed()),
    c_d3=Param(rand(1:100, 10, 10), Fixed()),
    c_d4=Param(randn(10, 10), Fixed()),
    # Arbitrary AbstractArrays
    c_e1=Param([[1.0, 2.0], zeros(3, 2, 1)], Fixed()),
    c_e2=Param([[1, 2], [1, 2]], Fixed()),
    c_e3=Param([[1.0, 2.0], [1.0, 2.0]], Fixed()),
    c_e4=Param([(rand(), rand()) for _ in 1:10], Fixed()),
    c_e5=Param([(rand(1:10), rand(1:10)) for _ in 1:10], Fixed()),
    ## Unconstrained Parameter
    # Floats
    u_a1=Param(1.0, Unconstrained()),
    u_a2=Param([2.0, 2.0], Unconstrained()),
    u_a3=Param([14.0 0.14; 0.14 14.0], Unconstrained()),
    u_b1=Param(Float32(26.0), Unconstrained()),
    u_b2=Param(Float16.([28.0, 28.0]), Unconstrained()),
    # Fixed buffer parameter
    u_d1=Param(rand(1:100, 100), Unconstrained()),
    u_d2=Param(randn(100), Unconstrained()),
    u_d3=Param(rand(1:100, 10, 10), Unconstrained()),
    u_d4=Param(randn(10, 10), Unconstrained()),
    # Arbitrary AbstractArrays
    u_e1=Param([[1.0, 2.0], zeros(3, 2, 1)], Unconstrained()),
    u_e2=Param([[1, 2], [1, 2]], Unconstrained()),
    u_e3=Param([[1.0, 2.0], [1.0, 2.0]], Unconstrained()),
    u_e4=Param([(rand(), rand()) for _ in 1:10], Unconstrained()),
    u_e5=Param([(rand(1:10), rand(1:10)) for _ in 1:10], Unconstrained()),
    ## Constrained Parameter -> Scalar only
    con_a1=Param(1.0, Constrained(0.0, 2.0)),
    con_b1=Param(Float32(26.0), Constrained(Float32(20.0), Float32(30.0))),
)

################################################################################
# Parameter for an example Model
_σ = Distributions.truncated(Distributions.Cauchy(10.0, 10), 0.0, 30.0)
_iwish2 = Distributions.InverseWishart(10.0, [1.0 0.0; 0.0 1.0])
_iwish3 = Distributions.InverseWishart(10.0, [1.0 0.0 0.0; 0 1 0; 0 0.0 1.0])
_ρ = [1.0 0.25; 0.25 1.0]
struct ExampleModel <: ModelName end
_val_examplemodel = (
    μ1=Param(1.0, Distributions.Normal()),
    μ2=Param([2.0, 3.0], Distributions.MvNormal(Diagonal(map(abs2, [1.0, 1.0])))),
    μ3=Param([4.0, 5.0], [Distributions.Normal(), Distributions.Normal()]),
    μ4=Param(
        [[6.0, 7.0], [8.0, 9.0]],
        [
            Distributions.MvNormal([2.0, 3.0], Diagonal(map(abs2, [1.0, 1.0]))),
            Distributions.MvNormal([1.0, 2.0], [3.0 0.4; 0.4 5.0]),
        ],
    ),
    σ1=Param(10.0, Distributions.Gamma()),
    σ3=Param([5.0, 5.0], [Distributions.Gamma(), Distributions.Gamma()]),
    σ2=Param([5.0, 5.0], [_σ, _σ]),
    ρ2=Param(copy(_ρ), Distributions.LKJ(2, 1.0)),
    σ4=Param([[5.0, 5.0], [5.0, 5.0]], [[_σ, _σ], [_σ, _σ]]),
    ρ4=Param([copy(_ρ), copy(_ρ)], [Distributions.LKJ(2, 1.0), Distributions.LKJ(2, 1.0)]),
    p=Param([0.2, 0.3, 0.5], Distributions.Dirichlet(3, 3.0)),
)
#_val_examplemodel_length = 23

################################################################################
# Parameter that may work in lower dimensions
struct LowerDims <: ModelName end
_val_lowerdims = (;
    ## Dirichlet
    ldim_e1=Param([0.1, 0.9], Distributions.Dirichlet(2, 2.0)),
    ldim_e2=Param(
        [[0.1, 0.9], [0.8, 0.2]],
        [Distributions.Dirichlet(2, 2.0), Distributions.Dirichlet(2, 2.0)],
    ),
    ldim_e3=Param([0.2, 0.3, 0.5], Distributions.Dirichlet(3, 3.0)),
    ldim_e4=Param(
        [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]],
        [Distributions.Dirichlet(3, 3.0), Distributions.Dirichlet(3, 3.0)],
    ),
    ## LKJ
    ldim_l1=Param([1.0 0.16; 0.16 1.0], Distributions.LKJ(2, 1.0)),
    ldim_l2=Param(
        [[1.0 0.20; 0.20 1.0], [1.0 0.24; 0.24 1.0]],
        [Distributions.LKJ(2, 1.0), Distributions.LKJ(2, 1.0)],
    ),
    ldim_l3=Param([1.0 0.16 0.16; 0.16 1.0 0.16; 0.16 0.16 1.0], Distributions.LKJ(3, 1.0)),
    ldim_l4=Param(
        [
            [1.0 0.16 0.16; 0.16 1.0 0.16; 0.16 0.16 1.0],
            [1.0 0.16 0.16; 0.16 1.0 0.16; 0.16 0.16 1.0],
        ],
        [Distributions.LKJ(3, 1.0), Distributions.LKJ(3, 1.0)],
    ),
    ## InverseWishart
    #        ldim_j1 = Param([15. .16 ; .16 18.], _iwish2),
    #        ldim_j2 = Param([[19. .20 ; .20 22.], [23. .24 ; .24 26.]],[_iwish2, _iwish2]),
    #        ldim_j3 = Param([15. .16 .16 ; .16 18. .16 ; .16 .16 20.], _iwish3),
    #        ldim_j4 = Param([[15. .16 .16 ; .16 18. .16 ; .16 .16 20.], [15. .16 .16 ; .16 18. .16 ; .16 .16 20.]],[_iwish3, _iwish3]),
)
