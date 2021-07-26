using AtomicGraphNets
using Test

@testset "Layers" begin
    include("layer_tests.jl")
end

@testset "Models" begin
    include("model_tests.jl")
end

@testset "Examples" begin
    include("example_tests.jl")
end
