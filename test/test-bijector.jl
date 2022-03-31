############################################################################################
# Bijectors
############################################################################################

@testset "Bijector - Simplex" begin
    simplex = [.3, .4, .3]
    bij = Bijectors.SimplexBijector()
    inv_bij = inverse(bij)
# Flatten
    _vals, unflat = flatten(simplex, bij)
    @test length(_vals) == length(simplex)
    _vals2 = unflat(_vals)
    @test size(_vals2) == size(simplex)
# Constraining
    _valsᵤ = constrain(bij, simplex)
    _vals3 = unconstrain(inv_bij, _valsᵤ)
    @test sum(_vals3) ≈ 1.0 atol = _TOL
end

@testset "Bijector - PDMatrix" begin
    covmat = [1. .5 ; .5 1.]
    bij = Bijectors.CorrBijector()
    inv_bij = inverse(bij)
# Flatten
    _vals, unflat = flatten(covmat, bij)
    @test length(_vals) == 1
    _vals2 = unflat(_vals)
    @test size(_vals2) == size(covmat)
# Constraining
    _valsᵤ = constrain(bij, covmat)
    #Lower triangular - hopefully fixed...
    @test _valsᵤ[1, 2] != 0.0
    _vals3 = unconstrain(inv_bij, _valsᵤ)
    @test all(_vals3 .≈ covmat)
end

@testset "Bijector - CorrMatrix" begin
    cormat = [1. .5 ; .5 1.]
    bij = Bijectors.CorrBijector()
    inv_bij = inverse(bij)
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
end
