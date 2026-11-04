import SwiftUI
import UIKit

/// The app's look: warm paper canvas, white cards with hairline borders, serif titles, dark ink pill buttons, one
/// accent (green) for "live". Light and dark palettes.
enum Theme {
    static func dynamic(_ light: UInt32, _ dark: UInt32) -> Color {
        Color(UIColor { $0.userInterfaceStyle == .dark ? rgb(dark) : rgb(light) })
    }
    static func rgb(_ h: UInt32) -> UIColor {
        UIColor(red: CGFloat(h >> 16 & 0xFF) / 255, green: CGFloat(h >> 8 & 0xFF) / 255, blue: CGFloat(h & 0xFF) / 255, alpha: 1)
    }

    static let canvas = dynamic(0xF8F7F4, 0x1C1C1A)
    static let surface = dynamic(0xFFFFFF, 0x2A2A27)
    static let sunken = dynamic(0xF0EFEB, 0x232321)
    static let hairline = dynamic(0xE4E3DE, 0x3A3A36)
    static let ink = dynamic(0x1E1E1C, 0xEDEDE8)
    static let muted = dynamic(0x75746E, 0x9C9B94)
    static let faint = dynamic(0xA9A8A1, 0x6E6D67)
    static let live = dynamic(0x2F9E5B, 0x4CC27B)
    static let stop = dynamic(0xD0473B, 0xE5675B)
    static let mark = dynamic(0xFFF1B8, 0x5A4B12)   // search-match highlight

    static func serif(_ size: CGFloat) -> Font { .system(size: size, weight: .regular, design: .serif) }
}

/// White card with a hairline border; the standard container.
struct CardStyle: ViewModifier {
    var padding: CGFloat = 14
    func body(content: Content) -> some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).strokeBorder(Theme.hairline))
    }
}

extension View {
    func card(padding: CGFloat = 14) -> some View { modifier(CardStyle(padding: padding)) }
    /// A screen: paper background behind lists and scroll views.
    func paper() -> some View { scrollContentBackground(.hidden).background(Theme.canvas.ignoresSafeArea()) }
    /// List row that shows its own card on the paper background.
    func cardRow() -> some View {
        listRowBackground(Color.clear).listRowSeparator(.hidden).listRowInsets(EdgeInsets(top: 5, leading: 16, bottom: 5, trailing: 16))
    }
}

/// Large serif page title with an optional caption, used instead of the system navigation title.
struct PageTitle: View {
    let title: String
    var caption: String? = nil
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            if let caption { Text(caption).font(.subheadline).foregroundStyle(Theme.muted) }
            Text(title).font(Theme.serif(34)).foregroundStyle(Theme.ink)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Small grey caption above a group of rows ("Today", "People").
struct GroupLabel: View {
    let text: String
    var body: some View { Text(text).font(.subheadline.weight(.medium)).foregroundStyle(Theme.muted) }
}

/// Rounded pill ("Start recording", filters).
struct Pill: View {
    let text: String
    var icon: String? = nil
    var selected = false
    var body: some View {
        HStack(spacing: 5) {
            if let icon { Image(systemName: icon).font(.caption.weight(.semibold)) }
            Text(text).lineLimit(1)
        }
        .font(.subheadline)
        .foregroundStyle(selected ? Theme.canvas : Theme.ink)
        .padding(.horizontal, 12).frame(height: 32)
        .background(selected ? Theme.ink : Theme.surface, in: Capsule())
        .overlay(Capsule().strokeBorder(selected ? .clear : Theme.hairline))
    }
}

/// Live audio bars.
struct LevelBars: View {
    var level: Float
    var active: Bool
    var body: some View {
        HStack(spacing: 3) {
            ForEach(0..<5, id: \.self) { i in
                let h = active ? CGFloat(5 + Double(level) * [12, 22, 16, 24, 10][i]) : 4
                Capsule().fill(active ? Theme.live : Theme.faint).frame(width: 3.5, height: h)
            }
        }
        .frame(height: 28)
        .animation(.easeOut(duration: 0.12), value: level)
    }
}
