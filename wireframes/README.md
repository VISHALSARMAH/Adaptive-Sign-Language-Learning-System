# ASL-Tutor Wireframes Documentation

This folder contains comprehensive wireframes and design documentation for the ASL-Tutor project.

## 📁 Contents

### User Interface Wireframes

1. **[01_landing_page.md](01_landing_page.md)**
   - Initial screen users see
   - Login/username input
   - Progress dashboard for returning users
   - Feature highlights
   - Accessibility controls

2. **[02_practice_session.md](02_practice_session.md)**
   - Active learning session interface
   - Target sign display
   - Webcam feed preview
   - Countdown timer
   - Session statistics
   - Sign progress tracking
   - Action buttons (Skip, Stop)

3. **[03_prediction_feedback.md](03_prediction_feedback.md)**
   - Correct prediction state
   - Incorrect prediction state
   - Low confidence warning state
   - Visual design elements
   - Color schemes and icons
   - Timing specifications

4. **[04_session_summary.md](04_session_summary.md)**
   - Session completion screen
   - Performance statistics
   - Per-sign breakdown table
   - Achievements display
   - AI-powered recommendations
   - Navigation options

### System Design

5. **[05_system_architecture.md](05_system_architecture.md)**
   - Complete system architecture diagram
   - Frontend layer (HTML/CSS/JavaScript)
   - Backend layer (FastAPI)
   - ML inference engine
   - Contextual bandit algorithm
   - Student model and data persistence
   - Training pipeline
   - Data flow diagram
   - Technology stack summary

6. **[06_mobile_responsive.md](06_mobile_responsive.md)**
   - Mobile portrait view (375×667)
   - Mobile landscape view
   - Tablet views (768×1024)
   - Responsive breakpoints
   - Touch interactions
   - Accessibility considerations

## 🎨 Design Principles

### Visual Hierarchy
- Large, clear target sign display
- High contrast feedback states
- Prominent action buttons
- Organized information cards

### User Experience
- Minimal friction for starting practice
- Instant visual feedback
- Clear progress indicators
- Motivational elements (achievements, streaks)

### Accessibility
- Audio toggle for text-to-speech
- High contrast mode
- Slow mode for extended timers
- Touch-friendly mobile interface

### Performance
- Real-time webcam processing
- Fast inference (50-100ms)
- Smooth animations
- Responsive across devices

## 🔧 Usage

### For Developers
- Reference these wireframes when implementing UI components
- Follow the specified element sizes and spacing
- Maintain consistent color schemes across states
- Implement responsive breakpoints as documented

### For Designers
- Use as foundation for high-fidelity mockups
- Maintain core layout structure
- Enhance visual design while preserving UX
- Test accessibility features thoroughly

### For Stakeholders
- Understand user flow and interactions
- Review feature placement and hierarchy
- Provide feedback on functionality
- Validate against requirements

## 📐 Key Measurements

### Desktop/Tablet
- Header: 60-80px height
- Webcam preview: 640×480px
- Buttons: 48px minimum height
- Cards: 12px border-radius, 0-4px shadows

### Mobile
- Buttons: 44-48px minimum (touch-friendly)
- Spacing: 8px between interactive elements
- Text: 16px minimum for body text
- Icons: 24-32px for primary actions

## 🎯 Component States

### Interactive Elements
- **Default**: Normal state
- **Hover**: Desktop cursor interaction
- **Active**: Touch/click state
- **Disabled**: Non-interactive state
- **Loading**: Processing state

### Feedback Colors
- ✅ **Success Green**: #4CAF50
- ❌ **Error Red**: #f44336
- ⚠️ **Warning Yellow**: #FFC107
- ℹ️ **Info Blue**: #2196F3

## 📱 Responsive Strategy

### Mobile-First Approach
1. Design for mobile (320px+)
2. Enhance for tablet (768px+)
3. Optimize for desktop (1024px+)

### Breakpoints
- Small mobile: 320px - 479px
- Large mobile: 480px - 767px
- Tablet: 768px - 1023px
- Desktop: 1024px+

## 🔄 User Flow

```
Landing → Enter Username → Camera Permission → 
Practice Session → Show Sign → Countdown → 
Capture → Predict → Feedback → Update Progress → 
Next Sign (loop) → Stop → Session Summary → 
Return to Dashboard
```

## 🚀 Implementation Priority

### Phase 1: MVP
- [x] Landing page
- [x] Practice session
- [x] Prediction feedback
- [x] Basic session summary

### Phase 2: Enhanced Features
- [ ] Detailed analytics dashboard
- [ ] User profile management
- [ ] Learning history graphs
- [ ] Social features (leaderboard)

### Phase 3: Advanced
- [ ] Mobile app (React Native)
- [ ] Offline mode
- [ ] Advanced visualizations
- [ ] Gamification features

## 📚 Related Documentation

- **README.md**: Main project documentation
- **GRAPHS/**: Training visualizations
- **docs/SCREENSHOTS.md**: Actual application screenshots
- **FUTURE_UPDATES.md**: Planned enhancements

## 💡 Design Tips

1. **Keep it simple**: Minimize cognitive load
2. **Provide feedback**: Always show system state
3. **Be consistent**: Use same patterns throughout
4. **Test accessibility**: Screen readers, keyboard nav
5. **Optimize performance**: Fast loading, smooth animations

---

**Last Updated**: February 6, 2026  
**Version**: 1.0.0  
**Author**: Vishal Sarmah
