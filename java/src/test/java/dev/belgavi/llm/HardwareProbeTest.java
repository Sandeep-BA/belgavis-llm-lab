// Author: Sandeep Belgavi
package dev.belgavi.llm;

import static org.junit.jupiter.api.Assertions.*;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import org.junit.jupiter.api.Test;

class HardwareProbeTest {

    @Test
    void probeNeverThrowsAndReportsCpu() {
        HardwareProbe.Report r = HardwareProbe.probe();
        assertTrue(r.cpuCores() >= 1);
        assertTrue(r.ramBytes() >= 0);
        assertEquals(r.cudaAvailable(), !r.cudaDevices().isEmpty());
        assertTrue(List.of("GPU", "CPU+", "CPU").contains(r.tier()));
    }

    @Test
    void jsonHasSharedKeys() throws Exception {
        JsonNode n = new ObjectMapper().readTree(HardwareProbe.toJson(HardwareProbe.probe()));
        for (String k : List.of("os", "arch", "cpuCores", "ramBytes", "cudaAvailable",
                "cudaDevices", "mpsAvailable", "tier")) {
            assertTrue(n.has(k), k);
        }
    }

    @Test
    void appleSiliconDetection() {
        assertTrue(HardwareProbe.isAppleSilicon("Mac OS X", "aarch64"));
        assertFalse(HardwareProbe.isAppleSilicon("Mac OS X", "x86_64"));
        assertFalse(HardwareProbe.isAppleSilicon("Linux", "aarch64"));
    }

    @Test
    void tierRules() {
        assertEquals("GPU", HardwareProbe.tier(List.of("GPU 0: x"), true));
        assertEquals("CPU+", HardwareProbe.tier(List.of(), true));
        assertEquals("CPU", HardwareProbe.tier(List.of(), false));
    }
}
