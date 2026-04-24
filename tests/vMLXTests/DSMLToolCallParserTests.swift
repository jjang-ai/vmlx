// SPDX-License-Identifier: Apache-2.0
//
// §385 DeepSeek V4 Flash DSML tool-call parser.
//
// Reference: `jang-tools/jang_tools/dsv4/test_chat.py::parse_dsml_tool_calls`
// and `research/DSV4-RUNTIME-ARCHITECTURE.md` §4.3.
//
// DSML envelope uses the fullwidth bar character U+FF5C (｜) around the
// marker "｜DSML｜". Parameter values carry a `string="true"|"false"`
// attribute — `false` means the value must be JSON-parsed (so "5" → 5,
// "true" → true, "[1,2]" → [1,2]). The parser must preserve both code
// paths bit-for-bit compatible with the Python reference.

import XCTest
@testable import vMLXEngine

final class DSMLToolCallParserTests: XCTestCase {
    private let p = DSMLToolCallParser()
    private let pfx = "｜DSML｜"  // fullwidth U+FF5C bars

    /// Single invoke with one string param.
    func testSimpleStringInvoke() {
        let out = """
        I'll search for that.
        <\(pfx)invoke name="search">
          <\(pfx)parameter name="query" string="true">weather today</\(pfx)parameter>
        </\(pfx)invoke>
        """
        let r = p.extractToolCalls(out, request: nil)
        XCTAssertTrue(r.toolsCalled)
        XCTAssertEqual(r.toolCalls.count, 1)
        XCTAssertEqual(r.toolCalls[0].name, "search")
        XCTAssertTrue(r.toolCalls[0].arguments.contains("\"weather today\""))
        XCTAssertEqual(r.content, "I'll search for that.")
    }

    /// Non-string (JSON-parsed) param coerces "5" → 5, "true" → true.
    func testJSONParamCoercion() {
        let out = """
        <\(pfx)invoke name="fetch_top">
          <\(pfx)parameter name="count" string="false">5</\(pfx)parameter>
          <\(pfx)parameter name="verbose" string="false">true</\(pfx)parameter>
          <\(pfx)parameter name="tags" string="false">["a","b"]</\(pfx)parameter>
        </\(pfx)invoke>
        """
        let r = p.extractToolCalls(out, request: nil)
        XCTAssertTrue(r.toolsCalled)
        XCTAssertEqual(r.toolCalls[0].name, "fetch_top")
        // args is a JSON blob; parse and verify types.
        let data = r.toolCalls[0].arguments.data(using: .utf8)!
        let parsed = try! JSONSerialization.jsonObject(with: data) as! [String: Any]
        XCTAssertEqual(parsed["count"] as? Int, 5)
        XCTAssertEqual(parsed["verbose"] as? Bool, true)
        XCTAssertEqual(parsed["tags"] as? [String], ["a", "b"])
    }

    /// Multiple consecutive invokes in one output.
    func testMultipleInvokes() {
        let out = """
        <\(pfx)invoke name="one">
          <\(pfx)parameter name="k" string="true">v1</\(pfx)parameter>
        </\(pfx)invoke>
        <\(pfx)invoke name="two">
          <\(pfx)parameter name="k" string="true">v2</\(pfx)parameter>
        </\(pfx)invoke>
        """
        let r = p.extractToolCalls(out, request: nil)
        XCTAssertTrue(r.toolsCalled)
        XCTAssertEqual(r.toolCalls.count, 2)
        XCTAssertEqual(r.toolCalls[0].name, "one")
        XCTAssertEqual(r.toolCalls[1].name, "two")
    }

    /// Reasoning block is stripped before invoke regex so a `<|DSML|invoke>`
    /// that shows up literally inside `<think>...</think>` won't false-match.
    func testReasoningBlockStripped() {
        let out = """
        <think>
        Let me think about using <\(pfx)invoke name="fake">...
        </think>
        <\(pfx)invoke name="real">
          <\(pfx)parameter name="x" string="true">ok</\(pfx)parameter>
        </\(pfx)invoke>
        """
        let r = p.extractToolCalls(out, request: nil)
        XCTAssertEqual(r.toolCalls.count, 1)
        XCTAssertEqual(r.toolCalls[0].name, "real")
    }

    /// No DSML markers → toolsCalled=false, content preserved verbatim.
    func testNoToolCall() {
        let out = "The answer is 4. Nothing to call here."
        let r = p.extractToolCalls(out, request: nil)
        XCTAssertFalse(r.toolsCalled)
        XCTAssertTrue(r.toolCalls.isEmpty)
        XCTAssertEqual(r.content, out)
    }

    /// Parser is registered under both `dsml` and `deepseek_v4` aliases.
    func testParserRegistryAliases() {
        let dsml = ToolCallParserRegistry.make( "dsml")
        XCTAssertNotNil(dsml)
        XCTAssertTrue(dsml is DSMLToolCallParser)
        let aliased = ToolCallParserRegistry.make( "deepseek_v4")
        XCTAssertNotNil(aliased)
        XCTAssertTrue(aliased is DSMLToolCallParser)
    }
}
