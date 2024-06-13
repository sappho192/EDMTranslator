namespace EDMTranslator.Tokenization
{
    public interface ITokenizer
    {
        public (int[], int[]) Encode(string sentence);
        public string Decode(uint[] tokens);
    }
}
