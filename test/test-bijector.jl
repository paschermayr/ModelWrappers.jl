############################################################################################
# Bijectors
############################################################################################
@testset "Bijector - Simplex" begin
    d = Distributions.Dirichlet(3,3)
    simplex = [.3, .4, .3]
    bij = Bijectors.SimplexBijector()
    inv_bij = inverse(bij)
    @test ModelWrappers._to_bijector(d) == ModelWrappers._to_bijector(bij)
    @test ModelWrappers._checkparam(simplex, bij)
# Flatten
    _vals, unflat = flatten(simplex, bij)
    @test length(_vals) == length(simplex)-1
    _vals2 = unflat(_vals)
    @test size(_vals2) == size(simplex)
    @test all(_vals2 .≈ simplex)
# Constraining
    _valsᵤ = constrain(bij, simplex)
    _vals3 = unconstrain(inv_bij, _valsᵤ)
    @test sum(_vals3) ≈ 1.0 atol = _TOL
# Custom Simplex flattening
    _vals4 = ModelWrappers.flatten_Simplex(simplex)
    @test length(_vals4) == length(simplex)-1
    _vals5 = ModelWrappers.Simplex_from_flatten!(zeros(length(simplex)), _vals4)
    _vals6 = ModelWrappers.Simplex_from_flatten(_vals4)
    @test all(simplex .≈ _vals5)
    @test all(simplex .≈ _vals6)
# AD flatten
    _vals, unflat = flatten(FlattenDefault(;unflattentype = UnflattenAD()), simplex, bij)
    _vals_constrained = unflat(_vals)
    @test sum(_vals_constrained) ≈ 1.0 atol = _TOL
    θᵤ = randn(Float32, length(_vals))
    #!NOTE: In ModelWrapper, would constrain after this step
    _vals_constrained = unconstrain(inv_bij, unflat(θᵤ))
    @test sum(_vals_constrained) ≈ 1.0 atol = _TOL
    @test eltype(_vals_constrained) == eltype(θᵤ)
end


@testset "Bijector - PDMatrix" begin
    d = Distributions.InverseWishart(10.0, [1.2 .5 ; .5 3.4])
    covmat = [1.2 .5 ; .5 3.4]
    bij = Bijectors.PDBijector()
    inv_bij = inverse(bij)
    @test ModelWrappers._to_bijector(d) == ModelWrappers._to_bijector(bij)
    @test ModelWrappers._checkparam(covmat, bij)
# Flatten
    _vals, unflat = flatten(covmat, bij)
    @test length(_vals) == 3
    _vals2 = unflat(_vals)
    @test size(_vals2) == size(covmat)
    @test all(_vals2 .≈ covmat)
# Constraining
    _valsᵤ = constrain(bij, covmat)
    #Lower triangular - hopefully fixed...
    @test _valsᵤ[1, 2] == 0.0
    _vals3 = unconstrain(inv_bij, _valsᵤ)
    @test all(_vals3 .≈ covmat)
# Custom Matrix flattening
    _tag = ModelWrappers.tag(covmat, false, true)
    _vals4 = ModelWrappers.flatten_Symmetric(covmat, _tag)
    @test length(_vals4) == 3
# AD flatten
    _vals, unflat = flatten(FlattenDefault(;unflattentype = UnflattenAD()), covmat, bij)
    _vals_constrained = unflat(_vals)
    @test all(_vals_constrained .≈ covmat)
    θᵤ = randn(Float32, length(_vals))
    #!NOTE: In ModelWrapper, would constrain after this step
    _vals_constrained = unconstrain(inv_bij, unflat(θᵤ))
    @test eltype(_vals_constrained) == eltype(θᵤ)
end

@testset "Bijector - CorrMatrix" begin
    d = Distributions.LKJ(2, 1.0)
    cormat = [1. .5 ; .5 1.]
    bij = Bijectors.CorrBijector()
    inv_bij = inverse(bij)
    @test ModelWrappers._to_bijector(d) == ModelWrappers._to_bijector(bij)
    @test ModelWrappers._checkparam(cormat, bij)
# Flatten
    _vals, unflat = flatten(cormat, bij)
    @test length(_vals) == 1
    _vals2 = unflat(_vals)
    @test size(_vals2) == size(cormat)
# Constraining
    _valsᵤ = constrain(bij, cormat)
    #Upper triangular
    @test _valsᵤ[1, 2] != 0.0
    _vals3 = unconstrain(inv_bij, _valsᵤ)
    @test all(_vals3 .≈ cormat)
# AD flatten
    _vals, unflat = flatten(FlattenDefault(;unflattentype = UnflattenAD()), cormat, bij)
    _vals_constrained = unflat(_vals)
    @test all(_vals_constrained .≈ cormat)
    θᵤ = randn(Float32, length(_vals))
    #!NOTE: In ModelWrapper, would constrain after this step
    _vals_constrained = unconstrain(inv_bij, unflat(θᵤ))
    @test eltype(_vals_constrained) == eltype(θᵤ)
end
