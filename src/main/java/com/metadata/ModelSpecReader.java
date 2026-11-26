package com.metadata;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ModelSpecReader {
    private static final Path MODEL_FILE = Paths.get("data", "models", "spec.json");
    private final ObjectMapper mapper = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);

    public ModelSpec read() throws IOException {
        if (!Files.exists(MODEL_FILE)) {
            throw new FileNotFoundException(
                    "Model spec not found: " + MODEL_FILE.toAbsolutePath()
            );
        }
        return mapper.readValue(MODEL_FILE.toFile(), ModelSpec.class);
    }

    public void write(ModelSpec spec) throws IOException {
        Files.createDirectories(MODEL_FILE.getParent());
        mapper.writeValue(MODEL_FILE.toFile(), spec);
    }
}
