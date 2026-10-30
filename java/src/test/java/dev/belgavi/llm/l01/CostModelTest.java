// Author: Sandeep Belgavi
package dev.belgavi.llm.l01;

import static org.junit.jupiter.api.Assertions.*;

import dev.belgavi.llm.l01.CostModel.ContextWindow;
import dev.belgavi.llm.l01.CostModel.Pricing;
import dev.belgavi.llm.l01.CostModel.RunCost;
import dev.belgavi.llm.l01.CostModel.Step;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;

class CostModelTest {

    @Test
    void pricingCost() {
        Pricing p = new Pricing(2.0, 10.0);
        assertEquals(2.0, p.cost(1_000_000, 0), 1e-9);
        assertEquals(2.0, p.cost(500_000, 100_000), 1e-9);
    }

    @Test
    void contextWindowFitAndPercent() {
        ContextWindow w = new ContextWindow(1000, 200);
        assertEquals(800, w.usableForInput());
        assertTrue(w.fits(800));
        assertFalse(w.fits(801));
        assertEquals(25.0, w.percentUsed(250), 1e-9);
    }

    @Test
    void turnsUntilFull() {
        ContextWindow w = new ContextWindow(1000, 100);
        assertEquals(12, CostModel.turnsUntilFull(w, 300, 50));
        assertEquals(0, CostModel.turnsUntilFull(w, 950, 50));
        assertThrows(IllegalArgumentException.class, () -> CostModel.turnsUntilFull(w, 0, 0));
    }

    @Test
    void agentRunResendsHistoryEveryCall() {
        RunCost run = CostModel.agentRunCost(
                100, List.of(new Step(10, 40), new Step(20, 0)), new Pricing(1_000_000, 1_000_000), null);
        assertEquals(List.of(100L, 150L), run.perCallInput());
        assertEquals(250, run.inputTokens());
        assertEquals(30, run.outputTokens());
        assertEquals(170, run.peakContextTokens());
        assertEquals(280.0, run.cost(), 1e-9);
    }

    @Test
    void agentRunInputGrowsFasterThanSteps() {
        List<Step> steps = Collections.nCopies(10, new Step(20, 200));
        RunCost run = CostModel.agentRunCost(500, steps, new Pricing(1, 1), null);
        assertTrue(run.inputTokens() > 10 * 500 * 2);
    }

    @Test
    void agentRunReportsWindowOverflow() {
        List<Step> steps = Collections.nCopies(3, new Step(10, 900));
        assertFalse(CostModel.agentRunCost(100, steps, new Pricing(1, 1), new ContextWindow(1000, 0)).fitsWindow());
    }

    @Test
    void agentRunRejectsBadInput() {
        assertThrows(IllegalArgumentException.class,
                () -> CostModel.agentRunCost(10, List.of(), new Pricing(1, 1), null));
    }

    @Test
    void sampleLanguagesAreComplete() {
        for (String lang : Samples.LANGUAGES) {
            assertEquals(4, Samples.sentences(lang).size());
            assertFalse(Samples.paragraph(lang).isBlank());
        }
    }
}
