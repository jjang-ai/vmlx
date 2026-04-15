// Copyright © 2026 Apple Inc.

import Foundation

extension JSONDecoder {
    public static func json5() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.allowsJSON5 = true
        return decoder
    }
}
