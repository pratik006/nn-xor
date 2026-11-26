package com.metadata;

import com.fasterxml.jackson.annotation.JsonInclude;

@JsonInclude(JsonInclude.Include.NON_NULL)
public record ModelSpec(
        int specVersion,
        String name,
        String description,
        Architecture architecture,
        Training training,
        Inference inference
) {
    public ModelSpec() {
        this(
                1,
                "unnamed-model",
                "",
                new Architecture(),
                new Training(),
                new Inference()
        );
    }
}
