namespace EDMTranslator
{
    public interface ITokenizer
    {
        public (int[], int[]) Encode(string input);
        public string Decode(uint[] input);
    }
}
