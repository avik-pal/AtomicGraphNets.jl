using Serialization
using ChemistryFeaturization
using AtomicGraphNets
using FastDEQ
using SteadyStateDiffEq
using OrdinaryDiffEq

inputs = deserialize.(readdir("examples/2_deq/data/inputs/", join=true))

struct InnerBlock{L1,L2}
    layer1::L1
    layer2::L2
end

Flux.@functor InnerBlock

function (ib::InnerBlock)(lapl, feat1, feat2)
    return lapl, ib.layer1(lapl, feat1)[2] .+ ib.layer2(lapl, feat2)[2]
end

model = DeepEquilibriumNetwork(
    InnerBlock(AGNConv(61 => 61), AGNConv(61 => 61)),
    DynamicSS(Tsit5(); abstol = 1e-3, reltol = 1e-3),
    maxiters = 100,
)

model(inputs[1])

Flux.gradient(() -> sum(model(inputs[1])[2]), Flux.params(model)).grads