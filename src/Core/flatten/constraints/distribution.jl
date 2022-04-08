#!NOTE: Make Distribution.Distributions work in ModelWrappers and map a Bijector to each distribution.
############################################################################################
function construct_transform(d::Distributions.Distribution, val)
    transform = Bijectors.bijector(d)
    return transform, Bijectors.inverse(transform)
end

############################################################################################
function _check(
    _rng::Random.AbstractRNG,
    d::Distributions.Distribution,
    val::Union{R,Array{R},AbstractArray},
) where {R<:Real}
    _val = rand(_rng, d)
    return typeof(val) == typeof(_val) && size(val) == size(_val) ? true : false
end

############################################################################################
# Implicit constraint through density by specialization -> else just flatten
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    constraint::Distributions.Distribution,
    x::Union{R,Array{R}},
) where {
    T<:AbstractFloat,
    F<:FlattenTypes,
    U<:UnflattenTypes,
    R<:Real,
}
    return construct_flatten(T, flattentype, unflattentype, x)
end

############################################################################################
#=
!NOTE: For this constraint, we use a Bijector as Transformer, so we do not need to add a new functors
1. MyTransformer <: AbstractTransformer, MyInverseTransformer <: AbstractTransformer
2. define a function construct_transform(MyConstraint, val) -> MyTransformer, MyInverseTransformer
3. overload unconstrain and log_abs_det_jac on MyTransformer.
4. overload constrain on MyInverseTransformer.
=#

############################################################################################
#Export
export
    construct_flatten,
    construct_transform,
    constrain,
    unconstrain,
    log_abs_det_jac
