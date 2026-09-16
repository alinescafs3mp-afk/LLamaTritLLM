using System.Text.Json;
namespace TritStudio.Core;

public static class CommandEnvelope
{
    // Unknown *well-formed* commands still receive a failed completion from Handle.
    // An invalid envelope has no trustworthy correlation identity: close the channel, never ignore it.
    public static WorkerCommand Parse(string line)
    {
        using var document = JsonDocument.Parse(line);
        var root = document.RootElement;
        if (root.ValueKind != JsonValueKind.Object) throw new InvalidDataException("Command must be an object.");
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var property in root.EnumerateObject())
            if (!names.Add(property.Name)) throw new InvalidDataException("Duplicate command field.");
        var command = root.Deserialize<WorkerCommand>(JsonData.Options);
        if (command is null || string.IsNullOrWhiteSpace(command.Kind) || command.Kind.Length > 64 ||
            string.IsNullOrWhiteSpace(command.Id) || command.Id.Length > 128 ||
            command.Kind.Any(char.IsControl) || command.Id.Any(char.IsControl) ||
            command.Payload.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("Incomplete or invalid command envelope.");
        return command;
    }
}
