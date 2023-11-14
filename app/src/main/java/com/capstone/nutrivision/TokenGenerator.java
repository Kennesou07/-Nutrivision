package com.capstone.nutrivision;

import java.util.UUID;

public class TokenGenerator {

    public static String generateToken() {
        // Generate a random UUID
        UUID uuid = UUID.randomUUID();

        // Convert UUID to a string (remove hyphens if needed)
        String token = uuid.toString().replace("-", "");

        return token;
    }

    public static void main(String[] args) {
        // Example: Generate and print a random token
        String token = generateToken();
        System.out.println("Generated Token: " + token);
    }
}
