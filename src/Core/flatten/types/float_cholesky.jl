function isapprox(x::Cholesky, y::Cholesky)
    return isapprox(x.factors, y.factors)
end

############################################################################################
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenStrict,
    x::Cholesky
) where {T<:AbstractFloat,F<:FlattenTypes}
    len_lower = binomial(size(x, 1), 2)
    len = length(x.factors)
    sz = size(x)
    buffer_flat = zeros(T, len)
    function flatten_to_Cholesky(v::Cholesky)
        ArgCheck.@argcheck binomial(size(v, 1), 2) == len_lower
        return fill_array!(buffer_flat, v.factors)
    end
    R = eltype(x.factors)
    buffer_unflat = zeros(R, sz)
    function unflatten_to_Cholesky(v::Union{R,AbstractVector{R}}) where {R<:Real}
        ArgCheck.@argcheck length(v) == len
        return Cholesky(fill_array!(buffer_unflat, v), 'L', 0)
    end
    return flatten_to_Cholesky, unflatten_to_Cholesky
end

function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenFlexible,
    x::Cholesky
) where {T<:AbstractFloat,F<:FlattenTypes}
    len_lower = binomial(size(x, 1), 2)
    len = length(x.factors)
    sz = size(x)
    function flatten_to_Cholesky(v::Cholesky)
        ArgCheck.@argcheck binomial(size(v, 1), 2) == len_lower
        return fill_array!(zeros(eltype(v.factors), len), v.factors)
    end
    function unflatten_to_Cholesky(v::Union{R,AbstractVector{R}}) where {R<:Real}
        ArgCheck.@argcheck length(v) == len
        return Cholesky(fill_array!(zeros(eltype(v), sz), v), 'L', 0)
    end
    return flatten_to_Cholesky, unflatten_to_Cholesky
end

############################################################################################
#Export
export
    construct_flatten
