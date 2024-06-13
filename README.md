- [EDMTranslator](#edmtranslator)
- [Nuget Package list](#nuget-package-list)
- [Requirements](#requirements)
- [Supported models](#supported-models)
- [Quickstart](#quickstart)
  - [Install the packages](#install-the-packages)
  - [Prepare the required data](#prepare-the-required-data)
    - [Japanese dictionary](#japanese-dictionary)
    - [Fine-tuned translator model](#fine-tuned-translator-model)
  - [Implement the driver code](#implement-the-driver-code)
- [How to build](#how-to-build)

# EDMTranslator

Text translator library based on LLM models, especially EncoderDecoderModel in HuggingFace

# Nuget Package list

| Package       | repo                                                                                                                            | description  |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| EDMTranslator | [![Nuget EDMTranslator](https://img.shields.io/nuget/v/EDMTranslator.svg?style=flat)](https://www.nuget.org/packages/EDMTranslator/) | Main library |

# Requirements

* .NET 6 or above

# Supported models

* FF14Translator([sappho192/ffxiv-ja-ko-translator](https://github.com/sappho192/ffxiv-ja-ko-translator)): Japanese-to-Korean translator based on `bert-base-japanese` and `skt-kogpt2-base-v2`
* More to be added...

# Quickstart

Following guide supposes that you are to use FF14Translator mentioned above.

## Install the packages

1. From the NuGet, install `EDMTranslator` package
2. And then, install `Tokenizers.DotNet.runtime.win` package too

## Prepare the required data

### Japanese dictionary

* Download unidic mecab dictionary `unidic-mecab-2.1.2_bin.zip` from [https://clrd.ninjal.ac.jp/unidic_archive/cwj/2.1.2/]() and unzip the archive into somewhere

### Fine-tuned translator model

* Download the translator model from [sappho192/ffxiv-ja-ko-translator/releases](https://github.com/sappho192/ffxiv-ja-ko-translator/releases/tag/0.2.1) and unzip the archive into somewhere

## Implement the driver code

Write the code like below and you are good to go 🫡
Note that you need to fix the path of `encoderDictDir` and `modelDir` correctly.

```csharp
 // Console application which translates Japanese sentence to Korean based on FF14Translator

using EDMTranslator.Tokenization;
using EDMTranslator.Translation;

// Prepare the tokenizer
var encoderVocabPath = await BertJapaneseTokenizer.HuggingFace.GetVocabFromHub("tohoku-nlp/bert-base-japanese-v2");
var hubName = "skt/kogpt2-base-v2";
var decoderVocabFilename = "tokenizer.json";
var decoderVocabPath = await Tokenizers.DotNet.HuggingFace.GetFileFromHub(hubName, decoderVocabFilename, "deps");

string encoderDictDir = @"D:\DATASET\unidic-mecab-2.1.2_bin";
var tokenizer = new BertJa2KoGPTTokenizer(
    encoderDictDir: encoderDictDir, encoderVocabPath: encoderVocabPath,
    decoderVocabPath: decoderVocabPath);

void TestTokenizer(ITokenizer tokenizer)
{
    Console.WriteLine("--Tokenizer test--");
    Console.WriteLine("[Encode]");
    var sentenceJa = "打ち合わせが終わった後にご飯を食べましょう。";
    Console.WriteLine($"Input: {sentenceJa}");
    var (embeddingsJa, attentionMask) = tokenizer.Encode(sentenceJa);
    Console.WriteLine($"Encoded: {string.Join(", ", embeddingsJa)}");

    Console.WriteLine("[Decode]");
    // Tokens of "음, 이제 식사도 해볼까요"
    var tokens = new uint[] { 9330, 387, 12857, 9376, 18649, 9098, 7656, 6969, 8084, 1 };
    Console.WriteLine($"Input: {string.Join(", ", tokens)}");
    var decoded = tokenizer.Decode(tokens);
    Console.WriteLine($"Decoded: {decoded}");
}
TestTokenizer(tokenizer);

// Prepare the translator
string modelDir = @"D:\MODEL\ffxiv-ja-ko-translator\onnx"; // Contains encoder_model.onnx and decoder_model_merged.onnx
var translator = new FF14Translator(tokenizer, modelDir);
void TestTranslator(FF14Translator translator)
{
    Console.WriteLine("--Translator test--");
    Translate(translator, "打ち合わせが終わった後にご飯を食べましょう。");
    Translate(translator, "試験前に緊張したあまり、熱がでてしまった。");
    Translate(translator, "山田は英語にかけてはクラスの誰にも負けない。");
    Translate(translator, "この本によれば、最初の人工橋梁は新石器時代にさかのぼるという。");
}
TestTranslator(translator);

static void Translate(FF14Translator translator, string sentence)
{
    Console.WriteLine($"SourceText: {sentence}");
    string translated = translator.Translate(sentence);
    Console.WriteLine($"Translated: {translated}");
}
```

# How to build

1. Prepare following stuff:
   1.  .NET build system (`dotnet 6.0`)
   2.  PowerShell (Recommend `7.4.2` or above)
2. Run `cbuild.ps1`

The build artifact will be saved in `nuget` directory.  
