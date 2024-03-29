#unflat = unflattenmethods[1]
#flat = flattentypes[1]
#floattypes = outputtypes[1]
############################################################################################
@testset "Flattening - Probabilistic Models - flatten type consistency" begin
    @test _checkparams(val_dist)
    _params = merge(val_dist, val_dist_nested)
    ## Iterate trough all Params in TestHelper.jl file
    for sym in eachindex(_params)
        #println(sym)
        param = _params[sym]
        θ = _get_val(param)
        constraint = _get_constraint(param)
        ## Check all flatten possibilities
        for unflat in unflattenmethods
            for flat in flattentypes
                for floattypes in outputtypes
                  # println(unflat, " ", flat, " ", floattypes)
                    flatdefault = FlattenDefault(;
                        output = floattypes,
                        flattentype = flat,
                    )
                    _flatten, _unflatten = construct_flatten(flatdefault, unflat, constraint, θ)
                    θ_flat = _flatten(θ)
                    θ_unflat = _unflatten(θ_flat)



                    #!NOTE Do not test if FlattenContinuous and empty Integer Param struct is evaluated
                    if θ_flat isa Vector{T} where {T<:AbstractFloat}
                        if unflat isa UnflattenStrict
                            @test eltype(θ_flat) == floattypes
                            @test typeof(θ_unflat) == typeof(θ)
                        end
                        if unflat isa UnflattenFlexible && θ_unflat isa Array{T} where {T<:AbstractFloat}
                            @test eltype(θ_unflat) == eltype(θ_flat)
                        end
                    end
                end
            end
        end
        @test _checkparams(param)
        @test _checkfinite(θ)
        @test _checkprior(constraint)
        _anyparam(_checksampleable(constraint))
        @test _allparam(_checksampleable(constraint))

        transformer = TransformConstructor(constraint, θ)
        ## Type Check 1 - Flatten/Unflatten
        _flatten, _unflatten = construct_flatten(FlattenDefault(), UnflattenStrict(), constraint, θ)
        _flattenAD, _unflattenAD = construct_flatten(FlattenDefault(), UnflattenFlexible(), constraint, θ)
        θ_flat = _flatten(θ)
        θ_unflat = _unflatten(θ_flat)
        @test typeof(_unflatten(θ_flat)) == typeof(θ)
        ## Type Check 2 - Constrain/Unconstrain
        θ_unconstrained = unconstrain(transformer, θ)
        θ_constrained = constrain(transformer, θ_unconstrained)
        @test typeof(θ) == typeof(θ_constrained)
        ## Type Check 3 - size of flatten(constrained) == flatten(unconstrained) for current Bijectors
        _θ_vec1 = _flatten(θ)
        
        #Note: No longer valid as of Bijectors 0.13
        #       _θ_vec2 = _flatten(θ_unconstrained)
        #@test length(_θ_vec1) == length(_θ_vec2)
        #@test typeof(_unflatten(_θ_vec1)) == typeof(_unflatten(_θ_vec2))
        ## If applicable, check if gradients for supported AD frameworks can be computed
        if length(θ_flat) > 0
            reconstruct = ReConstructor(constraint, θ)
            transform = TransformConstructor(constraint, θ)
            info = ParameterInfo(FlattenDefault(), reconstruct, transform, θ)
            θ_flat_unconstrained = unconstrain_flatten(info, θ)
            function check_AD_closure(constraint, val)
                reconstruct = ReConstructor(constraint, val)
                transform = TransformConstructor(constraint, val)
                info = ParameterInfo(FlattenDefault(), reconstruct, transform, val)
                function check_AD(θₜ::AbstractVector{T}) where {T<:Real}
                    θ = unflattenAD_constrain(info, θₜ)
                    return log_prior(constraint, θ) + log_abs_det_jac(transformer, θ)
                end
            end
            check_AD = check_AD_closure(constraint, θ)
            check_AD(θ_flat_unconstrained)
            grad_mod_fd = ForwardDiff.gradient(check_AD, θ_flat_unconstrained)
            grad_mod_rd = ReverseDiff.gradient(check_AD, θ_flat_unconstrained)
            grad_mod_zy = Zygote.gradient(check_AD, θ_flat_unconstrained)[1]
            @test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
            @test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL
        end
    end
end

############################################################################################
@testset "Flattening - Non-Probabilistic Models - flatten type consistency" begin
    @test _checkparams(val_constrained)
    _params = merge(val_constrained)
    ## Iterate trough all Params in TestHelper.jl file
    for sym in eachindex(_params)
        #println(sym)
        param = _params[sym]
        θ = _get_val(param)
        constraint = _get_constraint(param)
        ## Check all flatten possibilities
        for unflat in unflattenmethods
            for flat in flattentypes
                for floattypes in outputtypes
                  # println(unflat, " ", flat, " ", floattypes)
                    flatdefault = FlattenDefault(;
                        output = floattypes,
                        flattentype = flat,
                    )
                    _flatten, _unflatten = construct_flatten(flatdefault, unflat, constraint, θ)
                    θ_flat = _flatten(θ)
                    θ_unflat = _unflatten(θ_flat)

                    #!NOTE Do not test if FlattenContinuous and empty Integer Param struct is evaluated
                    if !(constraint isa Fixed) && θ_flat isa Vector{T} where {T<:AbstractFloat}
                        if unflat isa UnflattenStrict
                            @test eltype(θ_flat) == floattypes
                            @test typeof(θ_unflat) == typeof(θ)
                        end
                        if unflat isa UnflattenFlexible && θ_unflat isa Array{T} where {T<:AbstractFloat}
                            @test eltype(θ_unflat) == eltype(θ_flat)
                        end
                    end
                end
            end
        end
        @test _checkparams(param)
        @test _checkfinite(θ)
        @test !_checkprior(constraint)
        _anyparam(_checksampleable(constraint))
        @test !_allparam(_checksampleable(constraint))
        transformer = TransformConstructor(constraint, θ)
        ## Type Check 1 - Flatten/Unflatten
        _flatten, _unflatten = construct_flatten(FlattenDefault(), UnflattenStrict(), constraint, θ)
        _flattenAD, _unflattenAD = construct_flatten(FlattenDefault(), UnflattenFlexible(), constraint, θ)
        θ_flat = _flatten(θ)
        θ_unflat = _unflatten(θ_flat)
        @test typeof(_unflatten(θ_flat)) == typeof(θ)
        ## Type Check 2 - Constrain/Unconstrain
        θ_unconstrained = unconstrain(transformer, θ)
        θ_constrained = constrain(transformer, θ_unconstrained)
        @test typeof(θ) == typeof(θ_constrained)
        ## Type Check 3 - size of flatten(constrained) == flatten(unconstrained) for current Bijectors
#        _θ_vec1 = _flatten(θ)
#        _θ_vec2 = _flatten(θ_unconstrained)
#        @test length(_θ_vec1) == length(_θ_vec2)
#        @test typeof(_unflatten(_θ_vec1)) == typeof(_unflatten(_θ_vec2))
        ## If applicable, check if gradients for supported AD frameworks can be computed
        if length(θ_flat) > 0
            reconstruct = ReConstructor(constraint, θ)
            transform = TransformConstructor(constraint, θ)
            info = ParameterInfo(FlattenDefault(), reconstruct, transform, θ)
            θ_flat_unconstrained = unconstrain_flatten(info, θ)
            function check_AD_closure(constraint, val)
                reconstruct = ReConstructor(constraint, val)
                transform = TransformConstructor(constraint, val)
                info = ParameterInfo(FlattenDefault(), reconstruct, transform, val)
                function check_AD(θₜ::AbstractVector{T}) where {T<:Real}
                    θ = unflattenAD_constrain(info, θₜ)
                    return log_prior(constraint, θ) + log_abs_det_jac(transformer, θ)
                end
            end
            check_AD = check_AD_closure(constraint, θ)
            check_AD(θ_flat_unconstrained)
            grad_mod_fd = ForwardDiff.gradient(check_AD, θ_flat_unconstrained)
            grad_mod_rd = ReverseDiff.gradient(check_AD, θ_flat_unconstrained)
            #!NOTE: Zygote would just record "Nothing" as gradient for Fixed/Unconstrained without a functor
            #grad_mod_zy = Zygote.gradient(check_AD, θ_flat_unconstrained)[1]
            @test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
            #@test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL
        end
    end
end

############################################################################################
@testset "Flattening - flatten type consistency" begin
    _temp = Param(Distributions.Normal{Float32}(Float32(1.0), Float32(2.0)), Float32(2.0), ) #In flatten case, everything would be returned as Float64 when unflattening
    _flatten, _unflatten = construct_flatten(FlattenDefault(), UnflattenStrict(), _temp.constraint, _temp.val)
    θ_flat = _flatten(_temp.val)
    θ_unflat = _unflatten(θ_flat)
    @test typeof(_temp.val) == typeof(θ_unflat)
end

############################################################################################
@testset "Flattening - discrete parameter are not flattened" begin
    _temp = (
        a=Param(Distributions.Normal(), 1.0),
        b=Param(Distributions.Categorical(40), 2),
        c=Param([Distributions.Normal(), Distributions.Normal(2.0, 3.0)], [3.0, 4.0], ),
        d=Param([Distributions.Categorical(10), Distributions.Categorical(10)], [5, 6], ),
    )
    _temp_val = _get_val(_temp)
    _temp_constraint = _get_constraint(_temp)
    _flatten, _unflatten = construct_flatten(FlattenDefault(), UnflattenStrict(), _temp_constraint, _temp_val)
    θ_flat = _flatten(_temp_val)
    θ_unflat = _unflatten(θ_flat)
    @test _allparam(_checksampleable(_temp_constraint))
    @test _anyparam(_checksampleable(_temp_constraint))
    @test _checkprior(_temp_constraint)

    @test length(θ_flat) == 3
    @test typeof(_unflatten(θ_flat)) == typeof(_temp_val)
end

############################################################################################
@testset "Flattening - discrete parameter flattened" begin
    df_all = FlattenDefault(Float64, FlattenAll())
    _temp = (
        a=Param(Distributions.Normal(), 1.0, ),
        b=Param(Distributions.Categorical(40), 2),
        c=Param([Distributions.Normal(), Distributions.Normal(2.0, 3.0)], [3.0, 4.0], ),
        d=Param([Distributions.Categorical(10), Distributions.Categorical(10)], [5, 6], ),
    )
    _temp_val = _get_val(_temp)
    _temp_constraint = _get_constraint(_temp)
    _flatten, _unflatten = construct_flatten(df_all, _temp_constraint, _temp_val)
    _val = _flatten(_temp_val)
    θ_unflat = _unflatten(_val)
    @test length(_val) == 6
    @test typeof(θ_unflat) == typeof(_temp_val)
end
