#pragma once
#include "ttsinfer.h"
#include <unordered_map>
#include <memory>
#include <stdexcept>

namespace ttsinfer::nn {

struct Parameter {
    Tensor tensor;
    bool requires_grad = false;

    Parameter() = default;
    Parameter(const Tensor& t, bool req_grad = false)
        : tensor(t), requires_grad(req_grad) {}
};

using StateDict = std::unordered_map<std::string, Parameter>;

class Module {
protected:
    std::unordered_map<std::string, Parameter> parameters_;
    std::unordered_map<std::string, std::unique_ptr<Module>> submodules_;

    virtual Tensor forward(const std::vector<Tensor>& inputs) = 0;

public:
    virtual ~Module() = default;

    template<typename... Args>
    Tensor operator()(Args&&... args) {
        std::vector<Tensor> inputs{std::forward<Args>(args)...};
        return forward(inputs);
    }

    void register_parameter(const std::string& name,
                            const Tensor& t,
                            bool requires_grad = false)
    {
        parameters_[name] = Parameter(t, requires_grad);
    }

    void register_module(const std::string& name,
                         std::unique_ptr<Module> m)
    {
        submodules_[name] = std::move(m);
    }

    void state_dict(StateDict& out, const std::string& prefix = "") {
        for (auto& [name, param] : parameters_) {
            out[prefix + name] = param;
        }
        for (auto& [name, module] : submodules_) {
            module->state_dict(out, prefix + name + ".");
        }
    }

    void load_state_dict(const StateDict& sd, bool strict = true) {
        for (auto& [name, param] : parameters_) {
            auto it = sd.find(name);
            if (it != sd.end()) {
                parameters_[name].tensor = it->second.tensor;
            } else if (strict) {
                throw std::runtime_error("Missing parameter: " + name);
            }
        }

        for (auto& [name, module] : submodules_) {
            module->load_state_dict(sd, strict);
        }
    }
};


class Linear : public Module {
    int in_features_;
    int out_features_;
    bool bias_;

public:
    Linear(int in_features, int out_features, bool bias = true)
        : in_features_(in_features),
          out_features_(out_features),
          bias_(bias)
    {
   
        Tensor W = Tensor::empty({in_features_, out_features_}, DType::F32);
        register_parameter("weight", W, true);

        if (bias_) {
            Tensor b = Tensor::empty({1, out_features_}, DType::F32);
            register_parameter("bias", b, true);
        }
    }

protected:
    Tensor forward(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 1)
            throw std::runtime_error("Linear expects 1 input");

        const Tensor& x = inputs[0];  // [batch, in_features]

        Tensor out = matmul(x, parameters_["weight"].tensor);

        if (bias_) {
            out = add(out, parameters_["bias"].tensor);
        }

        return out;
    }
};

} // namespace
