// Author: Sandeep Belgavi
package dev.belgavi.llm;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

/**
 * Reports CPU, RAM and GPU availability. Never throws: anything it cannot detect is reported as
 * absent (0 / empty / false). The Python twin is {@code belgavi_llm.hardware}; both print the same JSON keys.
 */
public final class HardwareProbe {

    /** Snapshot of the machine. Field names are the shared JSON keys. */
    public record Report(
            String os,
            String arch,
            int cpuCores,
            long ramBytes,
            boolean cudaAvailable,
            List<String> cudaDevices,
            boolean mpsAvailable,
            String tier) {}

    private HardwareProbe() {}

    public static Report probe() {
        String os = System.getProperty("os.name", "unknown");
        String arch = System.getProperty("os.arch", "unknown");
        List<String> cuda = cudaDevices();
        boolean mps = isAppleSilicon(os, arch);
        return new Report(os, arch, Runtime.getRuntime().availableProcessors(), totalRamBytes(),
                !cuda.isEmpty(), cuda, mps, tier(cuda, mps));
    }

    static boolean isAppleSilicon(String os, String arch) {
        String o = os.toLowerCase(Locale.ROOT);
        String a = arch.toLowerCase(Locale.ROOT);
        return o.contains("mac") && (a.equals("aarch64") || a.equals("arm64"));
    }

    static String tier(List<String> cudaDevices, boolean mps) {
        if (!cudaDevices.isEmpty()) return "GPU";
        return mps ? "CPU+" : "CPU";
    }

    static long totalRamBytes() {
        try {
            if (ManagementFactory.getOperatingSystemMXBean()
                    instanceof com.sun.management.OperatingSystemMXBean os) {
                return Math.max(0, os.getTotalMemorySize());
            }
        } catch (RuntimeException | LinkageError e) {
            // fall through: RAM is reported as 0 (unknown)
        }
        return 0;
    }

    /** Lists CUDA devices by running the fixed command {@code nvidia-smi -L}; empty if absent or failing. */
    static List<String> cudaDevices() {
        try {
            Process p = new ProcessBuilder("nvidia-smi", "-L").redirectErrorStream(true).start();
            String out = new String(p.getInputStream().readAllBytes());
            if (!p.waitFor(10, TimeUnit.SECONDS)) {
                p.destroyForcibly();
                return List.of();
            }
            if (p.exitValue() != 0) return List.of();
            return out.lines().filter(l -> l.startsWith("GPU ")).toList();
        } catch (IOException e) {
            return List.of();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return List.of();
        }
    }

    public static String toJson(Report r) {
        try {
            return new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT).writeValueAsString(r);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    public static void main(String[] args) {
        System.out.println(toJson(probe()));
    }
}
