############################################################################################
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenStrict,
    x::R
) where {T<:AbstractFloat,F<:FlattenTypes,R<:Real}
    flatten_to_Real(x::R) where {R<:Real} = T[x]
    unflatten_to_Real(v::Union{<:Real,AbstractVector{<:Real}}) = convert(R, only(v))
    return flatten_to_Real, unflatten_to_Real
end

function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenFlexible,
    x::R
) where {T<:AbstractFloat,F<:FlattenTypes,R<:Real}
    flatten_to_Real_AD(x::R) where {R<:Real} = [x]
    unflatten_to_Real_AD(v::Union{<:Real,AbstractVector{<:Real}}) = only(v)
    return flatten_to_Real_AD, unflatten_to_Real_AD
end

############################################################################################
#Export
export
    construct_flatten
