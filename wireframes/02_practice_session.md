# ASL-Tutor Practice Session Wireframe

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ╔═══════════════════════════════════════════════════════════════════╗ │
│  ║  🤟 ASL-TUTOR              Session: 00:05:23      [🔊] [◐] [🐢]   ║ │
│  ╚═══════════════════════════════════════════════════════════════════╝ │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  📊 Session Stats                                               │  │
│  │  Signs Practiced: 8  |  Accuracy: 87%  |  Avg Time: 2.4s      │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│                                                                         │
│     ┌───────────────────────────────────────────────────────┐         │
│     │                                                         │         │
│     │                    Practice Sign:                      │         │
│     │                                                         │         │
│     │                         [ H ]                           │         │
│     │                                                         │         │
│     │               (letter H in large font)                 │         │
│     │                                                         │         │
│     └───────────────────────────────────────────────────────┘         │
│                                                                         │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                                                                  │  │
│  │   ╔════════════════════════════════════════════════════╗        │  │
│  │   ║                                                    ║        │  │
│  │   ║                                                    ║        │  │
│  │   ║                                                    ║        │  │
│  │   ║                 📹 WEBCAM FEED                    ║        │  │
│  │   ║                                                    ║        │  │
│  │   ║              (640x480 video preview)              ║        │  │
│  │   ║                                                    ║        │  │
│  │   ║                                                    ║        │  │
│  │   ║                                                    ║        │  │
│  │   ╚════════════════════════════════════════════════════╝        │  │
│  │                                                                  │  │
│  │                    [Camera: Active ✅]                          │  │
│  │                                                                  │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                                                                  │  │
│  │         ⏱️  Get ready...  3                                     │  │
│  │                                                                  │  │
│  │         (Countdown timer)                                        │  │
│  │                                                                  │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│                                                                         │
│  ┌──────────────────────────────────────────┐                         │
│  │  Current Sign Progress                   │                         │
│  │  ──────────────────────────────────────  │                         │
│  │  Mastery: ███████░░░  72%               │                         │
│  │  Attempts: 12  |  Success Rate: 83%     │                         │
│  │  Avg Response: 2.1s                     │                         │
│  └──────────────────────────────────────────┘                         │
│                                                                         │
│                                                                         │
│  [  ⏭️  Skip Sign  ]          [  🛑  Stop Practice  ]                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Elements

### Top Header
- Session timer showing elapsed time
- Accessibility controls always visible
- Current session stats bar

### Sign Display Panel
- Large, clear display of the target sign
- High contrast for visibility
- Simple, focused design

### Webcam Feed
- Live video preview (640x480)
- Camera status indicator
- Clear borders to define the capture area

### Countdown/Feedback Area
- Shows countdown before capture (3, 2, 1...)
- Displays prediction results after capture:
  - "✓ Correct! Confidence: 98%" (green)
  - "✗ Incorrect. You showed: B (87%)" (red)
- Clear, large text for readability

### Sign Progress Panel
- Current mastery level for this sign
- Historical performance stats
- Visual progress bar

### Action Buttons
- **Skip Sign**: Move to next sign
- **Stop Practice**: End session and view summary
- Large, touch-friendly buttons

## Interaction Flow

1. **Display Target Sign** → User reads the sign to practice
2. **Countdown** → 3-2-1 preparation time
3. **Capture** → Automatic image capture from webcam
4. **Prediction** → AI processes and returns result
5. **Feedback** → Show correct/incorrect with confidence
6. **Update** → Update mastery and select next sign
7. **Repeat** → Continue with next sign
