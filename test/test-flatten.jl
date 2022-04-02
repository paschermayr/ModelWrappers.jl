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
            for flat in flattenmethods
                for floattypes in flattentypes
                    #println(unflat, " ", flat, " ", floattypes)
                    flatdefault = FlattenDefault(;
                        output = floattypes,
                        flattentype = flat,
                        unflattentype = unflat
                    )
                    θ_flat, _unflatten = flatten(flatdefault, θ, constraint)
                    θ_unflat = _unflatten(θ_flat)
                    #!NOTE Do not test if FlattenContinuous and empty Integer Param struct is evaluated
                    if flat isa FlattenAll || θ_flat isa Vector{T} where {T<:AbstractFloat}
                        @test eltype(θ_flat) == floattypes
                    end
                    #!NOTE: Check types if UnflattenStrict
                    if unflat isa UnflattenStrict
                        @test typeof(θ_unflat) == typeof(θ)
                    end
                end
            end
        end
        @test _checkparams(param)
        @test _checkfinite(θ)
        @test _checkprior(constraint)
        _anyparam(_checksampleable(constraint))
        @test _allparam(_checksampleable(constraint))
        bij = _to_bijector(constraint)
        bij⁻¹ = _to_inv_bijector(bij)
        ## Type Check 1 - Flatten/Unflatten
        θ_vec, unflat = flatten(df_strict, θ, constraint)
        @test typeof(unflat(θ_vec)) == typeof(θ)
        ## Type Check 2 - Constrain/Unconstrain
        θ_unconstrained = unconstrain(bij, θ)
        θ_constrained = constrain(bij⁻¹, θ_unconstrained)
        @test typeof(θ) == typeof(θ_constrained)
        ## Type Check 3 - size of flatten(constrained) == flatten(unconstrained) for current Bijectors
        _θ_vec1, _unflat1 = flatten(df_strict, θ, constraint)
        _θ_vec2, _unflat2 = flatten(df_strict, θ_unconstrained, constraint)
        @test length(_θ_vec1) == length(_θ_vec2)
        @test typeof(_unflat1(_θ_vec1)) == typeof(_unflat2(_θ_vec2))
        ## If applicable, check if gradients for supported AD frameworks can be computed
        if length(θ_vec) > 0
            function check_AD_closure(θ, constraint)
                bij = _to_bijector(constraint)
                bij⁻¹ = _to_inv_bijector(bij)
                _, _unflatten = flatten(df_AD, θ, constraint)
                return function check_AD(θₜ::AbstractVector{T}) where {T<:Real}
                    θ = _unflatten(θₜ)
                    θ = constrain(bij⁻¹, θ)
                    return log_prior(constraint, θ) + log_abs_det_jac(bij, θ)
                end
            end
            check_AD = check_AD_closure(θ, constraint)
            check_AD(θ_vec)
            grad_mod_fd = ForwardDiff.gradient(check_AD, θ_vec)
            grad_mod_rd = ReverseDiff.gradient(check_AD, θ_vec)
            grad_mod_zy = Zygote.gradient(check_AD, θ_vec)[1]
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
        for unflat in unflattenmethods
            for flat in flattenmethods
                for floattypes in flattentypes
                #    println(unflat, " ", flat, " ", floattypes)
                    flatdefault = FlattenDefault(;
                        output = floattypes,
                        flattentype = flat,
                        unflattentype = unflat
                    )
                    θ_flat, _unflatten = flatten(flatdefault, θ, constraint)
                    θ_unflat = _unflatten(θ_flat)
                    #!NOTE Do not test if FlattenContinuous and empty Integer Param struct is evaluated
                    if flat isa FlattenAll || θ_flat isa Vector{T} where {T<:AbstractFloat}
                        @test eltype(θ_flat) == floattypes
                    end
                    #!NOTE: Check types if UnflattenStrict
                    if unflat isa UnflattenStrict
                        @test typeof(θ_unflat) == typeof(θ)
                    end
                end
            end
        end
        @test _checkparams(param)
        @test _checkfinite(θ)
        bij = _to_bijector(constraint)
        bij⁻¹ = _to_inv_bijector(bij)
        ## Type Check 1 - Flatten/Unflatten
        θ_vec, unflat = flatten(df_strict, θ, constraint)
        @test typeof(unflat(θ_vec)) == typeof(θ)
        ## Type Check 2 - Constrain/Unconstrain
        θ_unconstrained = unconstrain(bij, θ)
        θ_constrained = constrain(bij⁻¹, θ_unconstrained)
        @test typeof(θ) == typeof(θ_constrained)
        ## Type Check 3 - size of flatten(constrained) == flatten(unconstrained) for current Bijectors
        _θ_vec1, _unflat1 = flatten(df_strict, θ, constraint)
        _θ_vec2, _unflat2 = flatten(df_strict, θ_unconstrained, constraint)
        @test length(_θ_vec1) == length(_θ_vec2)
        @test typeof(_unflat1(_θ_vec1)) == typeof(_unflat2(_θ_vec2))
        ## If applicable, check if gradients for supported AD frameworks can be computed
        if length(θ_vec) > 0
            function check_AD_closure(θ, constraint)
                bij = _to_bijector(constraint)
                bij⁻¹ = _to_inv_bijector(bij)
                _, _unflatten = flatten(df_AD, θ, constraint)
                return function check_AD(θₜ::AbstractVector{T}) where {T<:Real}
                    θ = _unflatten(θₜ)
                    θ = constrain(bij⁻¹, θ)
                    return log_prior(constraint, θ) + log_abs_det_jac(bij, θ)
                end
            end
            check_AD = check_AD_closure(θ, constraint)
            check_AD(θ_vec)
            grad_mod_fd = ForwardDiff.gradient(check_AD, θ_vec)
            grad_mod_rd = ReverseDiff.gradient(check_AD, θ_vec)
            @test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
            #!NOTE: Zygote would just record "Nothing" as gradient for Fixed/Unconstrained without a functor
            #grad_mod_zy = Zygote.gradient(check_AD, θ_vec)[1]
            #@test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL
        end
    end
end

############################################################################################
@testset "Flattening - flatten type consistency" begin
    _temp = Param(Float32(2.0), Distributions.Normal{Float32}(Float32(1.0), Float32(2.0))) #In flatten case, everything would be returned as Float64 when unflattening
    _val, _unflat = flatten(_temp.val, _temp.constraint)
    _temp_back = _unflat(_val)
    @test typeof(_temp.val) == typeof(_temp_back)
end

############################################################################################
@testset "Flattening - discrete parameter are not flattened" begin
    _temp = (
        a=Param(1.0, Distributions.Normal()),
        b=Param(2, Distributions.Categorical(40)),
        c=Param([3.0, 4.0], [Distributions.Normal(), Distributions.Normal(2.0, 3.0)]),
        d=Param([5, 6], [Distributions.Categorical(10), Distributions.Categorical(10)]),
    )
    _temp_val = _get_val(_temp)
    _temp_constraint = _get_constraint(_temp)
    _val, _unflat = flatten(_temp_val, _temp_constraint)
    @test _allparam(_checksampleable(_temp_constraint))
    @test _anyparam(_checksampleable(_temp_constraint))
    @test _checkprior(_temp_constraint)

    @test length(_val) == 3
    @test typeof(_unflat(_val)) == typeof(_temp_val)
end

############################################################################################
@testset "Flattening - discrete parameter flattened" begin
    df_all = FlattenDefault(Float64, FlattenAll(), UnflattenStrict())
    _temp = (
        a=Param(1.0, Distributions.Normal()),
        b=Param(2, Distributions.Categorical(40)),
        c=Param([3.0, 4.0], [Distributions.Normal(), Distributions.Normal(2.0, 3.0)]),
        d=Param([5, 6], [Distributions.Categorical(10), Distributions.Categorical(10)]),
    )
    _temp_val = _get_val(_temp)
    _temp_constraint = _get_constraint(_temp)
    _val, _unflat = flatten(df_all, _temp_val, _temp_constraint)
    @test length(_val) == 6
    @test typeof(_unflat(_val)) == typeof(_temp_val)
end
