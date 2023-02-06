############################################################################################
# Nested

############################################################################################
@testset "Nested - AbstractArray" begin
    val = [1., [2, 3], [4. 5. ; 6. 7.], 8., [9., 10.]]
#Default ReConstructor and flatten
    val_flat, _reconstruct = flatten(val)
    @test val_flat == flatten(_reconstruct, val)
    @test val == unflatten(_reconstruct, val_flat)
    unflattenAD(_reconstruct, val_flat)
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = [1., [2, 3], [4. 5. ; 6. 7.], 8., [9., 10.]]
            ReConstructor(val)
            reconstruct = ReConstructor(flatdefault, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == Float64 #Always most common type
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
#            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
#            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Constraint ~ Make Distribution as standard constraints are separately tested
            val = [1., [2, 3], [4. 5. ; 6. 7.], 8., [9., 10.]]
            constraint = [DistributionConstraint(Normal()), DistributionConstraint(MvNormal([1., 1.])), Fixed(), DistributionConstraint(Gamma()), Unconstrained()]
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == Float64
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
#            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
#            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Transforms
            val
            transformer = TransformConstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val)

            @test val_con ≈ val
            @test val_con isa typeof(val)
            @test val_uncon isa typeof(val)
            @test logabs isa AbstractFloat
            @test !ModelWrappers._check(_RNG, constraint, val)
        end
    end
end

@testset "Nested - AbstractArray - Automatic Differentiation" begin
    val = [1., [2., 3.], [4. 5. ; 6. 7.], 8., [9., 10.]]
    constraint = [DistributionConstraint(Normal()), DistributionConstraint(MvNormal([1., 1.])), Fixed(), DistributionConstraint(Gamma()), Unconstrained()]
    reconstruct = ReConstructor(constraint, val)
    val_flat = flatten(reconstruct, val)

    check_AD = check_AD_closure(constraint, val)
    check_AD(val_flat)
    grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat)
    grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat)
    grad_mod_zy = Zygote.gradient(check_AD, val_flat)[1]
#=
## Experimental
    _shadow = zeros(length(val_flat))
    grad_mod_enz = Enzyme.autodiff(check_AD,
        Enzyme.Duplicated(val_flat, _shadow)
    )
##
=#
    @test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL

    grad_mod_fd = ForwardDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_rd = ReverseDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_zy = Zygote.gradient(check_AD, flatten(reconstruct, val))[1]
#=
    _shadow = zeros(length(val_flat))
    grad_mod_enz = Enzyme.autodiff(check_AD,
        Enzyme.Duplicated(flatten(reconstruct, val), _shadow)
    )
=#
end

############################################################################################
@testset "Nested - Tuple" begin
    val = (1., [2, 3], [4. 5. ; 6. 7.], 8, [9., 10.])
#Default ReConstructor and flatten
    val_flat, _reconstruct = flatten(val)
    @test val_flat == flatten(_reconstruct, val)
    @test val == unflatten(_reconstruct, val_flat)
    unflattenAD(_reconstruct, val_flat)
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = (1., [2, 3], [4. 5. ; 6. 7.], 8, [9., 10.])
            ReConstructor(val)
            reconstruct = ReConstructor(flatdefault, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == Float64 #Always most common type
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
#            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
#            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Constraint ~ Make Distribution as standard constraints are separately tested
            val = (1., [2, 3], [4. 5. ; 6. 7.], 8, [9., 10.])
            constraint = (DistributionConstraint(Normal()), DistributionConstraint(MvNormal([1., 1.])), Fixed(), Fixed(), Unconstrained())
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == Float64
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
#            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
#            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Transforms
            transformer = TransformConstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val)

            @test all( val_con .== val )
            @test val_con isa typeof(val)
            @test val_uncon isa typeof(val)
            @test logabs isa AbstractFloat

            #!NOTE - Second constraint not correct -> would not be a ModelWrapper
            @test !ModelWrappers._check(_RNG, constraint, val)
        end
    end
end

@testset "Nested - Tuple - Automatic Differentiation" begin
    val = (1., [2., 3.], [4. 5. ; 6. 7.], 8., [9., 10.])
    constraint = (DistributionConstraint(Gamma(1,2)), DistributionConstraint(MvNormal([1., 1.])), Fixed(), Fixed(), Unconstrained())
    reconstruct = ReConstructor(constraint, val)
    val_flat = flatten(reconstruct, val)

    check_AD = check_AD_closure(constraint, val)
    check_AD(val_flat)
    grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat)
    grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat)
    grad_mod_zy = Zygote.gradient(check_AD, val_flat)[1]
## Experimental
    _shadow = zeros(length(val_flat))
    grad_mod_enz = Enzyme.autodiff(check_AD,
        Enzyme.Duplicated(val_flat, _shadow)
    )
##
    @test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - _shadow)) ≈ 0 atol = _TOL

    grad_mod_fd = ForwardDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_rd = ReverseDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_zy = Zygote.gradient(check_AD, flatten(reconstruct, val))[1]
    _shadow = zeros(length(val_flat))
    grad_mod_enz = Enzyme.autodiff(check_AD,
        Enzyme.Duplicated(flatten(reconstruct, val), _shadow)
    )
end

############################################################################################
@testset "Nested - NamedTuple" begin
    val = (a = Float16(1.0), b = [2, 3], c = [4. 5. ; 6. 7.], d = 8, e = [9., 10.], f = (g = (h = 3.)))
#Default ReConstructor and flatten
    val_flat, _reconstruct = flatten(val)
    @test val_flat == flatten(_reconstruct, val)
    @test val == unflatten(_reconstruct, val_flat)
    unflattenAD(_reconstruct, val_flat)
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = (a = Float16(1.0), b = [2, 3], c = [4. 5. ; 6. 7.], d = 8, e = [9., 10.], f = (g = (h = 3.)))
            ReConstructor(val)
            reconstruct = ReConstructor(flatdefault, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == Float64 #Always most common type
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
#            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
#            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Constraint ~ Make Distribution as standard constraints are separately tested
            val = (a = Float16(1.0), b = [2, 3], c = [4. 5. ; 6. 7.], d = 8, e = [9., 10.], f = (g = (h = 3.)))
            constraint = (a = DistributionConstraint(Normal()), b = DistributionConstraint(MvNormal([1., 1.])), c = Fixed(), d = Fixed(), e = Unconstrained(), f = (g = (h = DistributionConstraint(Gamma()))))
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == Float64
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
#            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
#            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Transforms
            transformer = TransformConstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val)

            @test val_con isa typeof(val)
            @test val_uncon isa typeof(val)
            @test logabs isa AbstractFloat
            #!NOTE - Second constraint not correct -> would not be a ModelWrapper
            @test !ModelWrappers._check(_RNG, constraint, val)
        end
    end
end

@testset "Nested - NamedTuple - Automatic Differentiation" begin
    val = (a = Float32(1.0), b = [2., 3.], c = [4. 5. ; 6. 7.], d = 8., e = [9., 10.], f = (g = (h = 3.)))
    constraint = (a = DistributionConstraint(Gamma(1,2)), b = DistributionConstraint(MvNormal([1., 1.])), c = Fixed(), d = Fixed(), e = Unconstrained(), f = (g = (h = DistributionConstraint(Gamma()))))
    reconstruct = ReConstructor(constraint, val)
    val_flat = flatten(reconstruct, val)

    check_AD = check_AD_closure(constraint, val)
    check_AD(val_flat)
    grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat)
    grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat)
    grad_mod_zy = Zygote.gradient(check_AD, val_flat)[1]
## Experimental
    _shadow = zeros(length(val_flat))
    grad_mod_enz = Enzyme.autodiff(check_AD,
        Enzyme.Duplicated(flatten(reconstruct, val), _shadow)
    )
##
    @test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - _shadow)) ≈ 0 atol = _TOL

    grad_mod_fd = ForwardDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_rd = ReverseDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_zy = Zygote.gradient(check_AD, flatten(reconstruct, val))[1]
    _shadow = zeros(length(val_flat))
    grad_mod_enz = Enzyme.autodiff(check_AD,
        Enzyme.Duplicated(flatten(reconstruct, val), _shadow)
    )
end
