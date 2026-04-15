#  Using a Model

Using a model is easy:  load the weights, tokenize and evaluate.

There is a high level API described in <doc:evaluation> and this documentation
describes the lower level API if you need more control.

## Loading a Model

A model is typically loaded by using a `ModelFactory` and a `ModelConfiguration`:

```swift
// e.g. LLMModelFactory.shared
let modelFactory: ModelFactory

// e.g. LLMRegistry.llama3_8B_4bit
let modelConfiguration: ModelConfiguration

// e.g. TokenizersLoader() from MLXLMTokenizers
let tokenizerLoader: any TokenizerLoader

let container = try await modelFactory.loadContainer(
    using: tokenizerLoader,
    configuration: modelConfiguration
)
```

The `container` provides an isolation context (an `actor`) to run inference in the model.

Predefined `ModelConfiguration` instances are provided as static variables
on the `ModelRegistry` types or they can be created:

```swift
let modelConfiguration = ModelConfiguration(id: "mlx-community/llama3_8B_4bit")
```

The flow inside the `ModelFactory` goes like this:

```swift
public class LLMModelFactory: ModelFactory {

    public func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> ModelContext {
        // modelDirectory and tokenizerDirectory are already resolved
        // load the base configuration
        // using the typeRegistry create a model (random weights)
        // load the weights, apply quantization as needed, update the model
            // calls model.sanitize() for weight preparation
        // load the tokenizer via tokenizerLoader.load(from: directory)
    }
}
```

Callers with specialized requirements can use these individual components to manually
load models, if needed.

## Evaluation Flow

- Load the Model
- UserInput
- LMInput
- generate()
    - NaiveStreamingDetokenizer
    - TokenIterator

## Evaluating a Model

Once a model is loaded you can evaluate a prompt or series of
messages. Minimally you need to prepare the user input:

```swift
let prompt = "Describe the image in English"
var input = UserInput(prompt: prompt, images: image.map { .url($0) })
input.processing.resize = .init(width: 256, height: 256)
```

This example shows adding some images and processing instructions -- if
model accepts text only then these parts can be omitted. The inference
calls are the same.

Assuming you are using a `ModelContainer` (an actor that holds
a `ModelContext`, which is the bundled set of types that implement a
model), the first step is to convert the `UserInput` into the
`LMInput` (LanguageModel Input):

```swift
let generateParameters: GenerateParameters
let input: UserInput

let result = try await modelContainer.perform { [input] context in
    let input = try context.processor.prepare(input: input)

```

Given that `input` we can call `generate()` to produce a stream
of tokens. In this example we use a `NaiveStreamingDetokenizer`
to assist in converting a stream of tokens into text and print it.
The stream is stopped after we hit a maximum number of tokens:

```
    var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)

    return try MLXLMCommon.generate(
        input: input, parameters: generateParameters, context: context
    ) { tokens in

        if let last = tokens.last {
            detokenizer.append(token: last)
        }

        if let new = detokenizer.next() {
            print(new, terminator: "")
            fflush(stdout)
        }

        if tokens.count >= maxTokens {
            return .stop
        } else {
            return .more
        }
    }
}
```

### Wired Memory (Optional)

Use the policy-based API to coordinate a single global wired limit across tasks.
`WiredMemoryManager` and `WiredMemoryTicket` are provided by MLX, while
MLXLMCommon adds LLM-oriented policies (like `WiredFixedPolicy` or capped sum).
Policy-only admission is enabled by default on unsupported backends so the same
ticket logic applies on CPU (no OS limit changes are attempted).

```swift
let policy = WiredSumPolicy()
let ticket = policy.ticket(size: estimatedBytes)

let stream = try MLXLMCommon.generate(
    input: input,
    parameters: generateParameters,
    context: context,
    wiredMemoryTicket: ticket
)
```

Policies are pure and compute a single limit for all active tickets. Built-in
policies include `WiredSumPolicy`, `WiredMaxPolicy`, and `WiredFixedPolicy`.
Use `WiredMemoryTicket.withWiredLimit` for cancellation-safe start/end pairing.

Policies can also gate concurrency by implementing `canAdmit`. When admission is
denied, `start()` suspends until capacity is available. For debugging, the
`WiredMemoryManager.events()` stream emits changes in DEBUG builds and is a no-op
in release builds.

If you want to account for long-lived model weights without keeping the wired
limit elevated while idle, create tickets with `kind: .reservation` so they
participate in admission and limit calculation only when active tickets exist.
