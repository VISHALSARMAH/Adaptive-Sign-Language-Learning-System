# ASL-Tutor Prediction Feedback States

## State 1: Correct Prediction

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ╔═══════════════════════════════════════════════════════════════╗ │
│  ║                                                               ║ │
│  ║                      ✅ CORRECT!                             ║ │
│  ║                                                               ║ │
│  ║              You successfully signed: H                       ║ │
│  ║                                                               ║ │
│  ║                  Confidence: 98.7%                            ║ │
│  ║                  Response Time: 1.8s                          ║ │
│  ║                                                               ║ │
│  ║              ⭐ Great job! Moving to next sign...            ║ │
│  ║                                                               ║ │
│  ╚═══════════════════════════════════════════════════════════════╝ │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Mastery Updated: ████████████░░░░░  75% (+3%)              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## State 2: Incorrect Prediction

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ╔═══════════════════════════════════════════════════════════════╗ │
│  ║                                                               ║ │
│  ║                      ❌ NOT QUITE                            ║ │
│  ║                                                               ║ │
│  ║              Target Sign: H                                   ║ │
│  ║              You showed: K (Confidence: 87.3%)                ║ │
│  ║                                                               ║ │
│  ║              Response Time: 3.2s                              ║ │
│  ║                                                               ║ │
│  ║              💡 Try again with this sign!                    ║ │
│  ║                                                               ║ │
│  ╚═══════════════════════════════════════════════════════════════╝ │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Don't worry! Let's practice this one more...               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## State 3: Low Confidence

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ╔═══════════════════════════════════════════════════════════════╗ │
│  ║                                                               ║ │
│  ║                      ⚠️ UNCLEAR                              ║ │
│  ║                                                               ║ │
│  ║              Couldn't recognize sign clearly                  ║ │
│  ║              Confidence: 24.3%                                ║ │
│  ║                                                               ║ │
│  ║              💡 Tips:                                        ║ │
│  ║              • Ensure good lighting                           ║ │
│  ║              • Keep hand in camera frame                      ║ │
│  ║              • Make sign clearly and hold steady             ║ │
│  ║                                                               ║ │
│  ║              [Try Again]                                      ║ │
│  ║                                                               ║ │
│  ╚═══════════════════════════════════════════════════════════════╝ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Visual Design Elements

### Colors
- **Correct (Green)**: #4CAF50 background, white text
- **Incorrect (Red)**: #f44336 background, white text
- **Low Confidence (Yellow)**: #FFC107 background, dark text

### Icons
- ✅ Checkmark for correct
- ❌ X mark for incorrect
- ⚠️ Warning for low confidence
- ⭐ Star for encouragement
- 💡 Lightbulb for tips

### Typography
- Large, bold status message
- Clear confidence percentage
- Readable response time
- Helpful feedback text

### Timing
- Display feedback for 2-3 seconds (normal mode)
- Display for 4-5 seconds (slow mode)
- Auto-advance on correct (optional manual advance)
- Manual advance on incorrect (practice again)
