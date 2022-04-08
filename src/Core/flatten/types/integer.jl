#!NOTE: Only methods for FlattenContinuous given here, so if all flattened, defaults to Real construct_flatten/unflatten
#!NOTE2: need to use unflattentype::UnflattenStrict/UnflattenFlexible instead of ::U so specializes correctly
############################################################################################
function construct_flatten(
    output::Type{T},
    flattentype::FlattenContinuous,
    unflattentype::UnflattenStrict,
    x::Union{I,Array{I}}
) where {T<:AbstractFloat, I<:Integer}
    v = I[]
    flatten_to_Int(x::Union{R,AbstractArray{R}}) where {R<:Real} = v
    unflatten_to_Int(v::Union{R,AbstractVector{R}}) where {R<:Real} = x
    return flatten_to_Int, unflatten_to_Int
end
function construct_flatten(
    output::Type{T},
    flattentype::FlattenContinuous,
    unflattentype::UnflattenFlexible,
    x::Union{I,Array{I}}
) where {T<:AbstractFloat, I<:Integer}
    v = I[]
    flatten_to_Int(x::Union{R,AbstractArray{R}}) where {R<:Real} = v
    unflatten_to_Int(v::Union{R,AbstractVector{R}}) where {R<:Real} = x
    return flatten_to_Int, unflatten_to_Int
end

############################################################################################
#Export
export
    construct_flatten
