############################################################################################
# types
#output = outputtypes[1]
#flattentype = flattentypes[1]
############################################################################################
@testset "Types - Float" begin
    val = Float16(1.0)
#Default ReConstructor and flatten
    val_flat, _reconstruct = flatten(val)
    @test val_flat == flatten(_reconstruct, val)
    @test val == unflatten(_reconstruct, val_flat)
    unflattenAD(_reconstruct, val_flat)
    for output in outputtypes
        for flattentype in flattentypes
            val = Float16(5.0)
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            ReConstructor(val)
            reconstruct = ReConstructor(flatdefault, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == typeof(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Constraint ~ Make Distribution as standard constraints are separately tested
            constraint_float = Distributions.Gamma(2,2)
            ReConstructor(constraint_float, val)
            reconstruct = ReConstructor(flatdefault, constraint_float, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == typeof(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test x_unflatAD2 isa eltype(x_flatAD)
################################################################################
# Transforms
            transformer =TransformConstructor(DistributionConstraint(constraint_float), val)
            val_uncon = ModelWrappers.unconstrain(transformer, val)
            val_con = ModelWrappers.constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val)

            @test val_con ≈ val
            @test val_con isa typeof(val)
            @test val_uncon isa typeof(val)
            @test logabs isa AbstractFloat
        end
    end
end

@testset "Types - Float - Automatic Differentiation" begin
    val = Float32(5.0)
    constraint = DistributionConstraint(Distributions.Gamma(2,2))
    reconstruct = ReConstructor(constraint, val)
    val_flat = flatten(reconstruct, val)

    check_AD = check_AD_closure(constraint, val)
    check_AD(val_flat)
    grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat)
    grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat)
    grad_mod_zy = Zygote.gradient(check_AD, val_flat)[1]
## Experimental
#    _shadow = zeros(length(val_flat))
#    grad_mod_enz = Enzyme.autodiff(check_AD,
#        Enzyme.Duplicated(val_flat, _shadow)
#    )
##
    @test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
    @test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL
#    @test sum(abs.(grad_mod_fd - _shadow)) ≈ 0 atol = _TOL

    grad_mod_fd = ForwardDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_rd = ReverseDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_zy = Zygote.gradient(check_AD, flatten(reconstruct, val))[1]
#    _shadow = zeros(length(val_flat))
#    grad_mod_enz = Enzyme.autodiff(check_AD,
#        Enzyme.Duplicated(flatten(reconstruct, val), _shadow)
#    )
end

############################################################################################
@testset "Types - Vector Float" begin
    val = Float16.([1., 2.])
#Default ReConstructor and flatten
    val_flat, _reconstruct = flatten(val)
    @test val_flat == flatten(_reconstruct, val)
    @test val == unflatten(_reconstruct, val_flat)
    unflattenAD(_reconstruct, val_flat)
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = Float16.([1., 2.])
            ReConstructor(val)
            reconstruct = ReConstructor(flatdefault, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test eltype(x_unflatAD) == eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test eltype(x_unflatAD2) == eltype(x_flatAD)
################################################################################
# Constraint ~ Make Distribution as standard constraints are separately tested
            constraint = DistributionConstraint(Distributions.MvNormal([1.,1.]))
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test eltype(x_unflatAD) == eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test eltype(x_unflatAD2) == eltype(x_flatAD)

################################################################################
# Check if errors if flattened parameter not correct size
            @test_throws ArgumentError flatten(reconstruct, zeros(length(val)+1))
            @test_throws ArgumentError flatten(reconstruct, zeros(length(val)-1))
            @test_throws ArgumentError flattenAD(reconstruct, zeros(length(val)+1))
            @test_throws ArgumentError flattenAD(reconstruct, zeros(length(val)-1))
            @test_throws ArgumentError unflatten(reconstruct, zeros(length(x_flat)+1))
            @test_throws ArgumentError unflatten(reconstruct, zeros(length(x_flat)-1))
            @test_throws ArgumentError unflattenAD(reconstruct, zeros(length(x_flat)+1))
            @test_throws ArgumentError unflattenAD(reconstruct, zeros(length(x_flat)-1))

################################################################################
# Transforms
            transformer = TransformConstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val)

            @test val_con ≈ val
            @test val_con isa typeof(val)
            @test val_uncon isa typeof(val)
            @test logabs isa AbstractFloat
            ModelWrappers._check(_RNG, constraint, val)
        end
    end
end

@testset "Types - Vector Float - Automatic Differentiation" begin
    val = [1., 2.]
    constraint = DistributionConstraint(Distributions.MvNormal([1., 1.]))
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
    @test sum(abs.(grad_mod_fd - _shadow)) ≈ 0 atol = _TOL
=#
##
    @test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
#    @test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL

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
@testset "Types - Array Float" begin
    val = Float16.([1. 0.3 ; .3 1.0])
#Default ReConstructor and flatten
    val_flat, _reconstruct = flatten(val)
    @test val_flat == flatten(_reconstruct, val)
    @test val == unflatten(_reconstruct, val_flat)
    unflattenAD(_reconstruct, val_flat)
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = Float16.([1. 0.3 ; .3 1.0])
            ReConstructor(val)
            reconstruct = ReConstructor(flatdefault, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test eltype(x_unflatAD) == eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test eltype(x_unflatAD2) == eltype(x_flatAD)

################################################################################
# Check if errors if flattened parameter not correct size
            @test_throws ArgumentError flatten(reconstruct, zeros(3,3))
            @test_throws ArgumentError flatten(reconstruct, zeros(1,1))
            @test_throws ArgumentError flattenAD(reconstruct, zeros(3,3))
            @test_throws ArgumentError flattenAD(reconstruct,  zeros(1,1))
            @test_throws ArgumentError unflatten(reconstruct, zeros(length(x_flat)+1))
            @test_throws ArgumentError unflatten(reconstruct, zeros(length(x_flat)-1))
            @test_throws ArgumentError unflattenAD(reconstruct, zeros(length(x_flat)+1))
            @test_throws ArgumentError unflattenAD(reconstruct, zeros(length(x_flat)-1))

################################################################################
# Constraint ~ Make Distribution as standard constraints are separately tested
            constraint = Unconstrained()
            ReConstructor(constraint, val)
            reconstruct = ReConstructor(flatdefault, constraint, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            @test eltype(x_flat) == output
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test eltype(x_unflatAD) == eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test eltype(x_unflatAD2) == eltype(x_flatAD)
################################################################################
# Transforms
            transformer = TransformConstructor(constraint, val)
            val_uncon = unconstrain(transformer, val)
            val_con = constrain(transformer, val_uncon)
            logabs = log_abs_det_jac(transformer, val)

            @test val_con ≈ val
            @test val_con isa typeof(val)
            @test val_uncon isa typeof(val)
            @test logabs isa AbstractFloat
            ModelWrappers._check(_RNG, constraint, val)
        end
    end
end

@testset "Types - Array Float - Automatic Differentiation" begin
    val = [1. 0.3 ; .3 1.0]
    constraint = DistributionConstraint(Distributions.Distributions.InverseWishart(10., [1. 0. ; 0. 1.]))
    reconstruct = ReConstructor(constraint, val)
    val_flat = flatten(reconstruct, val)

    check_AD = check_AD_closure(constraint, val)
    check_AD(val_flat)
    grad_mod_fd = ForwardDiff.gradient(check_AD, val_flat)
    #grad_mod_rd = ReverseDiff.gradient(check_AD, val_flat)
    grad_mod_zy = Zygote.gradient(check_AD, val_flat)[1]
#=
## Experimental
    _shadow = zeros(length(val_flat))
    grad_mod_enz = Enzyme.autodiff(check_AD,
        Enzyme.Duplicated(val_flat, _shadow)
    )
=#
##
    #@test sum(abs.(grad_mod_fd - grad_mod_rd)) ≈ 0 atol = _TOL
#    @test sum(abs.(grad_mod_fd - grad_mod_zy)) ≈ 0 atol = _TOL

    grad_mod_fd = ForwardDiff.gradient(check_AD, flatten(reconstruct, val))
    #grad_mod_rd = ReverseDiff.gradient(check_AD, flatten(reconstruct, val))
    grad_mod_zy = Zygote.gradient(check_AD, flatten(reconstruct, val))[1]
#=
    _shadow = zeros(length(val_flat))
    grad_mod_enz = Enzyme.autodiff(check_AD,
        Enzyme.Duplicated(flatten(reconstruct, val), _shadow)
    )
=#
end

############################################################################################
@testset "Types - Integer" begin
    val = Int16(1.0)
#Default ReConstructor and flatten
    val_flat, _reconstruct = flatten(val)
    @test val_flat == flatten(_reconstruct, val)
    @test val == unflatten(_reconstruct, val_flat)
    unflattenAD(_reconstruct, val_flat)
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = Int16(1.0)
            ReConstructor(val)
            reconstruct = ReConstructor(flatdefault, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            if flattentype isa FlattenAll
                @test eltype(x_flat) == output
            end
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test x_unflatAD isa eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test x_unflatAD2 isa eltype(x_flatAD)
        end
    end
end

############################################################################################
@testset "Types - Array Integer" begin
    val = Int16.([1 2 ; 3 4])
#Default ReConstructor and flatten
    val_flat, _reconstruct = flatten(val)
    @test val_flat == flatten(_reconstruct, val)
    @test val == unflatten(_reconstruct, val_flat)
    unflattenAD(_reconstruct, val_flat)
    for output in outputtypes
        for flattentype in flattentypes
            flatdefault = FlattenDefault(; output = output, flattentype = flattentype)
            val = Int16.([1 2 ; 3 4])
            ReConstructor(val)
            reconstruct = ReConstructor(flatdefault, val)
# Flatten
            x_flat = flatten(reconstruct, val)
            @test x_flat isa AbstractVector
            if flattentype isa FlattenAll
                @test eltype(x_flat) == output
            end
# Flatten AD
            x_flatAD = flattenAD(reconstruct, val)
            @test x_flatAD isa AbstractVector
            @test eltype(x_flatAD) == eltype(val)
# Unflatten
            x_unflat = unflatten(reconstruct, x_flat)
            @test x_unflat isa typeof(val)
# Unflatten AD
            x_unflatAD = unflattenAD(reconstruct, x_flat)
            @test eltype(x_unflatAD) == eltype(x_flat)
            x_unflatAD2 = unflattenAD(reconstruct, x_flatAD)
            @test eltype(x_unflatAD2) == eltype(x_flatAD)
        end
    end
end
