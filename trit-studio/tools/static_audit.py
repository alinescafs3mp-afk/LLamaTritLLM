#!/usr/bin/env python3
"""Structural source checks, NOT a C# compiler and NOT a UI/runtime test."""
from __future__ import annotations
import ast
import json
import pathlib
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    projects = list(ROOT.rglob('*.csproj'))
    for project in projects:
        tree = ET.parse(project)
        for item in tree.findall('.//ProjectReference'):
            assert (project.parent/item.attrib['Include']).resolve().is_file(), (project, item.attrib)
    for path in [ROOT/'Directory.Build.props', ROOT/'NuGet.Config']:
        ET.parse(path)
    solution = (ROOT/'TritStudio.sln').read_text()
    solution_projects = re.findall(r'"([^"\r\n]+\.csproj)"', solution)
    for project in solution_projects:
        assert (ROOT/project.replace('\\','/')).is_file(), project
    assert any('UiTests' in p for p in solution_projects)
    for file in ROOT.rglob('*.json'):
        if any(x in file.parts for x in ['.git','obj','bin','artifacts']): continue
        json.loads(file.read_text(encoding='utf-8-sig'))
    python_files = list((ROOT/'tools').glob('*.py'))
    for file in python_files: ast.parse(file.read_text(), filename=str(file))
    sh_files = list((ROOT/'scripts').glob('*.sh'))+list((ROOT/'packaging').glob('*.sh'))
    for file in sh_files: subprocess.run(['bash','-n',str(file)],check=True)
    core = (ROOT/'src/TritStudio.Core/ModelFiles.cs').read_text()
    assert 'using System.Runtime.InteropServices;' in core and 'using System.Security.Cryptography;' in core
    worker = (ROOT/'src/TritStudio.Trainer/Program.cs').read_text()
    assert 'CancelEpoch' in worker and 'Volatile.Read(ref _cancelEpoch)' in worker
    assert 'test.jsonl' not in worker, 'Blind test should not be automatically loaded into trainer.'
    assert worker.count('Dataset.LoadManyTraining(') == 2
    assert 'discard-pending' in worker and '_hasRuntimeRate' in worker
    assert 'BundledCorpus.Load' in worker
    store = (ROOT/'src/TritStudio.Trainer/WorkspaceStore.cs').read_text()
    for name in ['settings.json','base-train.json','validation.json','optimizer.bin','state.json','commit.json']:
        assert name in store
    ui = (ROOT/'src/TritStudio.App/MainWindow.cs').read_text()
    for token in ['RunButton','_busyButtons','mode-state','_actionTitle','_activity','_closing','onlineAfterCreate','UpdateContextBudget','MaintainSnapshots','ChatJournal.ReadTail','_windowLife']:
        assert token in ui
    tests = (ROOT/'tests/TritStudio.UiTests/Program.cs').read_text()
    assert 'Dispatch<int>(async' in tests and 'return 0;' in tests
    delivery = (ROOT/'tools/Delivery/Program.cs').read_text()
    assert 'ZipFile.CreateFromDirectory(folder, stagedZip' in delivery
    for name in ['WHERE_TO_PICK_UP.txt','DELIVERY_RESULT.json','trainer-cuda','win-x64','--self-test']:
        assert name in delivery
    cs_files = list((ROOT/'src').rglob('*.cs'))+list((ROOT/'tests').rglob('*.cs'))+list((ROOT/'tools/Delivery').glob('*.cs'))
    for file in cs_files:
        source = file.read_text()
        assert 'throw new NotImplementedException' not in source, file
        assert 'return 99;' not in source, file
        assert '<<<<<<<' not in source, file
    assert 'PruneWithUiLease' in (ROOT/'src/TritStudio.Core/WorkspaceMaintenance.cs').read_text()
    assert 'DisposeCoreAsync' in (ROOT/'src/TritStudio.App/WorkerClient.cs').read_text()
    training = (ROOT/'src/TritStudio.Trainer/TrainingSession.cs').read_text()
    for token in ['BatchPlanner','BeginEvaluation','_evaluatedVersion','Model.MarkUpdated','ScalarType.Float64','ValidationCacheHits']:
        assert token in training
    model = (ROOT/'src/TritStudio.Trainer/TorchModel.cs').read_text()
    assert 'scaled_dot_product_attention(q, k, v, null, 0.0, true)' in model and 'EndEvaluation' in model
    assert 'ReadInferenceRevision' in ui and 'LatestValueMailbox<string>' in ui and 'ExportGuard.Validate' in ui
    assert '--benchmark' in delivery
    assert 'fault-eof-alive' in (ROOT/'tests/TritStudio.UiTests/ClientFaultChecks.cs').read_text()
    assert 'Task.Run(ReadOutput)' in (ROOT/'src/TritStudio.App/WorkerClient.cs').read_text()
    assert 'BoundedTextCapture.ReadAsync' in (ROOT/'src/TritStudio.App/TrainerLocator.cs').read_text()
    assert 'ReadToEndAsync()' not in (ROOT/'src/TritStudio.App/TrainerLocator.cs').read_text()
    assert 'challenge' in (ROOT/'data/DATASET_MANIFEST.json').read_text()
    assert 'SupervisedBatch.Build' in training and 'ProjectOnlyTargets' in training
    assert 'ForwardForLoss' in training and 'RestoreState' in training
    assert 'targetPositions' in model and 'using var layerScope' in model
    assert 'reshape(-1, _c.GroupSize).clone()' in model and 'using var planeScope' in model
    client=(ROOT/'src/TritStudio.App/WorkerClient.cs').read_text()
    assert 'BoundedLineReader' in client and '_process.Kill(entireProcessTree: true)' in client
    assert 'new ValidationGuard' in worker and 'guard.EnsureTraining' in worker
    assert 'BoundedWriteStream' in core
    assert 'AllowsWithinBudget' in (ROOT/'src/TritStudio.Core/ManagedInference.cs').read_text()
    assert 'Audit5Checks.All' in (ROOT/'tests/TritStudio.Tests/Program.cs').read_text()
    assert 'HashingWriteStream' in core and 'WriteHashed' in core
    quantizer=(ROOT/'src/TritStudio.Core/TernaryQuantizer.cs').read_text()
    assert 'Span<float> residual = stackalloc float[128]' in quantizer and 'weights.Clone()' not in quantizer
    assert 'SquaredGradientNorm' in training and 'using var gradientScope' in training
    assert 'guard.EnsureTraining(pending, ct)' in worker and 'OnlineReplay.Build' in worker
    assert 'guard.EnsureTraining([row], ct)' in (ROOT/'src/TritStudio.Core/OnlineReplay.cs').read_text()
    assert 'RetainedHistoryStart' in (ROOT/'src/TritStudio.Core/ValidationGuard.cs').read_text()
    assert 'Audit6Checks.All' in (ROOT/'tests/TritStudio.Tests/Program.cs').read_text()
    manifest=json.loads((ROOT/'data/DATASET_MANIFEST.json').read_text())
    assert manifest['version'] in (ROOT/'src/TritStudio.Core/BundledCorpus.cs').read_text()
    assert 'challenge-v6.jsonl' in (ROOT/'src/TritStudio.Core/BundledCorpus.cs').read_text()
    assert 'LearningStages.CreationOptions' in worker and 'CreationMode.Untrained' in ui
    assert 'LoadSeed("pretrain.jsonl", ct)' in worker and '_online = false; _paused = true;' in worker
    assert 'StageArchive.Preserve' in worker and 'fixedContext' in ui and 'learningHistory' in ui
    assert 'nn.functional.silu(gate)' in model
    assert 'new byte[Math.Min(1024' in (ROOT/'src/TritStudio.Core/ManagedInference.cs').read_text()
    assert '17.0.0-audit17' in (ROOT/'VERSION').read_text() and '17.0.0-audit17' in (ROOT/'Directory.Build.props').read_text()
    assert 'аудит 17' in ui and 'RU v5' not in ui
    assert client.index('CommandCompletion.Parse(message.Data)') < client.index('_pending.TryRemove(id, out var pending)')
    assert 'EvaluationBatchCache' in training and 'RetainedPayloadBytes' in (ROOT/'src/TritStudio.Core/EvaluationBatchCache.cs').read_text()
    rollback = worker[worker.index('    private void Rollback()'):]
    assert rollback.index('LoadWorkspace(path, announce: false, reconcile: false)') < rollback.index('JsonData.AtomicWrite(FileAt("active.json")')
    assert 'ValidationBaseline.Anchor' in worker and 'ValidationSequenceLength' in store
    assert 'ValidateMaterialRequest' in worker and 'PreserveWithReceipt' in worker
    assert 'Audit8Checks.All' in (ROOT/'tests/TritStudio.Tests/Program.cs').read_text()
    assert 'challenge-v8.jsonl' in (ROOT/'src/TritStudio.Core/BundledCorpus.cs').read_text()
    assert 'CheckpointJsonCache' in store and 'trainingHash, validationHash' in store
    assert 'CheckpointStateGuard.Validate' in worker
    assert 'StrictDatasetLines.Read' in (ROOT/'src/TritStudio.Core/Dataset.cs').read_text()
    assert 'File.ReadLines' not in (ROOT/'src/TritStudio.Core/Dataset.cs').read_text()
    assert 'Не удалось сохранить переписку' in ui
    assert '_targetCounts' in (ROOT/'src/TritStudio.Core/BatchPlanner.cs').read_text()
    assert 'Audit9Checks.All' in (ROOT/'tests/TritStudio.Tests/Program.cs').read_text()
    assert 'EncodeInto' in (ROOT/'src/TritStudio.Core/ByteTokenizer.cs').read_text()
    assert 'var input = new int[n]' in (ROOT/'src/TritStudio.Core/Dataset.cs').read_text()
    assert 'if (!_needsRecovery) return' in worker and 'Режим не сохранён' in worker
    assert '_writeTimeout' in client and 'WriteLineAsync(json.AsMemory(), ct)' in client
    assert 'CheckWriteDeadline' in (ROOT/'tests/TritStudio.UiTests/ClientFaultChecks.cs').read_text()
    assert 'CheckModeFailureAndRefusal' in (ROOT/'tests/TritStudio.Tests/WorkerProtocolChecks.cs').read_text()
    assert 'Audit10Checks.All' in (ROOT/'tests/TritStudio.Tests/Program.cs').read_text()
    assert 'CommandEnvelope.Parse(line)' in worker and 'Duplicate command field' in (ROOT/'src/TritStudio.Core/CommandEnvelope.cs').read_text()
    assert 'Dataset.ReuseUnchanged' in worker and 'ReferenceEquals(all, _trainingData)' in worker
    assert '_store.EnsureCapacity' in worker and 'CheckpointBudget.Publications' in worker
    assert 'CancelPendingModeEdit' in ui and 'epoch != _epoch' in ui and 'allowOwnedOperation: true' in ui
    assert 'shutdownWrite.ContinueWith' in client
    assert 'Audit11Checks.All' in (ROOT/'tests/TritStudio.Tests/Program.cs').read_text()
    assert 'CheckMalformedCommandChannel' in (ROOT/'tests/TritStudio.Tests/WorkerProtocolChecks.cs').read_text()
    assert 'echo-mode' in (ROOT/'tests/TritStudio.UiTests/ClientFaultChecks.cs').read_text()
    assert 'ReplayPreparationBenchmark.Run' in (ROOT/'src/TritStudio.Trainer/TrainerBenchmark.cs').read_text()
    assert 'Audit12Checks.All' in (ROOT/'tests/TritStudio.Tests/Program.cs').read_text()
    cpu = (ROOT/'src/TritStudio.Core/CpuProjection.cs').read_text()
    inference = (ROOT/'src/TritStudio.Core/ManagedInference.cs').read_text()
    assert 'CpuProjection.Triple' in inference and 'CpuProjection.Pair' in inference
    assert 'RowsPerChunk = 16' in cpu and 'ManagedInference.Session.Dot' in cpu
    assert 'denominators' in inference and 'denominators' in model
    assert 'EventEnvelope.Parse(line)' in client and 'completedKind != expected.Kind' in client
    assert client.index('EventEnvelope.CompletionCommand(message.Data)') < client.index('_pending.TryRemove(id, out var pending)')
    assert 'UniqueFields(data)' in (ROOT/'src/TritStudio.Core/EventEnvelope.cs').read_text()
    assert 'info.Revision != publication.Info.Revision' in ui
    assert '_snapshotLoadSignature == signature' in ui and 'await inFlight.Task' in ui
    assert '_modelLoader.WaitAsync(load.Token)' in ui and 'CancelSnapshotLoad' in ui
    assert 'fault-wrong-command' in (ROOT/'tests/TritStudio.UiTests/ClientFaultChecks.cs').read_text()
    assert 'Repeated publication reported completion' in tests
    assert 'first == 0 ? full' in (ROOT/'src/TritStudio.Core/ValidationGuard.cs').read_text()
    assert 'CpuPreparationBenchmark.Run' in (ROOT/'src/TritStudio.Trainer/TrainerBenchmark.cs').read_text()
    assert 'conversation-ru-v17' == manifest['version']
    assert 'challenge-v12.jsonl' in (ROOT/'src/TritStudio.Core/BundledCorpus.cs').read_text()
    assert 'DraftRecovery.Restore' in ui and 'RestoreFailedDraft(input, excluded)' in ui
    assert 'CpuAttention.Compute' in inference
    journal = (ROOT/'src/TritStudio.Core/ChatJournal.cs').read_text()
    assert 'long end = file.Length' in journal and 'while (read < wanted)' in journal
    assert 'turns.Count < limit' in journal and 'StrictUtf8.GetCharCount' in journal
    assert 'AttentionJournalBenchmark.Run' in (ROOT/'src/TritStudio.Trainer/TrainerBenchmark.cs').read_text()
    assert 'Audit13Checks.All' in (ROOT/'tests/TritStudio.Tests/Program.cs').read_text()
    assert 'challenge-v13.jsonl' in (ROOT/'src/TritStudio.Core/BundledCorpus.cs').read_text()
    assert 'WorkspacePreview.Load' in ui and 'ApplyHistory(preview.History)' in ui
    opener = ui[ui.index('    private async Task OpenWorkspaceCore'):ui.index('    private void OnWorker')]
    assert opener.index('WorkspacePreview.Load') < opener.index('long epoch = ++_epoch') < opener.index('_client.DisposeAsync')
    assert 'if (!installed && !same) nextLease.Dispose()' in opener
    assert 'CpuElementwise.RmsNorm' in inference and 'CpuElementwise.AddInPlace' in inference
    assert 'CapturePublicationMaster' in store and '_initialMaster = null; // BEFORE' in training
    assert training.index('_initialMaster = null; // BEFORE') < training.index('Optimizer.step()')
    assert 'Audit14Checks.All' in (ROOT/'tests/TritStudio.Tests/Program.cs').read_text()
    assert 'InitializationBenchmark.Run' in (ROOT/'src/TritStudio.Trainer/TrainerBenchmark.cs').read_text()
    library_ui=(ROOT/'src/TritStudio.App/ModelLibraryUi.cs').read_text()
    assert 'ModelLibrary.MoveToTrash' in library_ui and 'await DisconnectSelectedModel()' in library_ui
    assert 'CurrentModelPath => _workspace ?? _inferenceFile' in library_ui
    assert 'SelectActualModel();' in library_ui and 'ClearModelSpecificInputs();' in ui
    assert 'if (r.Config is not null) ApplyPreset(r.Config);' not in ui
    assert 'LayoutTransformControl' in ui and 'SetSidebar' in library_ui
    delivery=(ROOT/'tools/Delivery/Program.cs').read_text()
    assert 'bool update = !args.Contains("--full")' in delivery
    assert 'UpdatePayloadPolicy.IsOwnedPayload' in delivery and 'REQUIRED_RUNTIME_FILES.json' in delivery
    assert 'UPDATE_SHA256SUMS.json' in delivery and '64L*1024*1024' in delivery
    assert 'Audit15Checks.All' in (ROOT/'tests/TritStudio.Tests/Program.cs').read_text()
    assert 'Audit15UiChecks.Run' in (ROOT/'tests/TritStudio.UiTests/Program.cs').read_text()
    assert 'Button("Очистить чат")' in ui and 'RunButton(newChat, "Очистка чата", ClearChat)' in ui
    clear = ui[ui.index('    private void ClearConversation()'):ui.index('    private async Task<bool> Confirm')]
    assert clear.index('ChatSessionState.StartNew') < clear.index('_history.Clear()')
    assert 'Invoke("ApplyHistory",ChatJournal.ReadTail' in (ROOT/'tests/TritStudio.UiTests/Audit15UiChecks.cs').read_text()
    assert 'mean(new long[] { 1 }' in model and 'mean(new long[] { -1 }' in model
    assert 'if (exception is InvalidDataException)' in client and 'new IOException(exception.Message, exception)' in client
    assert 'CloseStdoutPipeEnds' in (ROOT/'tests/TritStudio.UiTests/ClientFaultChecks.cs').read_text()
    assert 'REQUIRED_RUNTIME_FILES.json' in (ROOT/'packaging/Check-Update.ps1').read_text()
    assert '--cpu-only --full' not in (ROOT/'scripts/build.sh').read_text()
    assert 'RoutingStrategies.Tunnel' in ui and '_composerEnterDown' in ui
    assert '_input.KeyDown +=' not in ui
    assert 'KeyPress(Key.Enter' in (ROOT/'tests/TritStudio.UiTests/Audit16UiChecks.cs').read_text()
    assert 'PrefixFinalBlockSkips' in inference and '_optimizePrefix && !computeLogits' in inference
    assert 'PrefixBenchmark.Run' in (ROOT/'src/TritStudio.Trainer/TrainerBenchmark.cs').read_text()
    assert 'Directory.EnumerateDirectories(revisions, "r*")' in core
    assert 'ActiveRevision>(pointer, 4096)' in core
    assert 'Test-LaptopChecks.ps1' in (ROOT/'packaging/Check-laptop.ps1').read_text()
    assert 'rid != "win-x64"' in delivery and 'exact-deployed' not in ui
    assert 'challenge-v16.jsonl' in (ROOT/'src/TritStudio.Core/BundledCorpus.cs').read_text()
    assert 'ValidateInitialization' in training and 'ValidateForTraining' in training
    assert 'ModelPanels' in ui and all(name in ui for name in ['ActiveModelPanel','CreateModelPanel','TrainModelPanel'])
    assert 'CreationResources(config)' in ui and 'CreationTraining()' in ui
    assert 'LearningSmoke.Run' in (ROOT/'src/TritStudio.Trainer/TrainerSelfTest.cs').read_text()
    assert 'CheckCreationBudgetAndQuality' in (ROOT/'tests/TritStudio.Tests/WorkerProtocolChecks.cs').read_text()
    assert 'QualityProbe.Run' in worker
    differences = subprocess.run(['git','diff','--check'],cwd=ROOT,text=True,capture_output=True)
    assert differences.returncode == 0, differences.stdout+differences.stderr
    staged = subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,text=True,capture_output=True)
    assert staged.returncode == 0, staged.stdout+staged.stderr
    core_tests = len(re.findall(r'^    \("', (ROOT/'tests/TritStudio.Tests/Program.cs').read_text(), re.M))
    core_tests += sum(len(re.findall(r'^        \("', (ROOT/name).read_text(), re.M)) for name in ['tests/TritStudio.Tests/Audit5Checks.cs','tests/TritStudio.Tests/Audit6Checks.cs','tests/TritStudio.Tests/Audit7Checks.cs','tests/TritStudio.Tests/Audit8Checks.cs','tests/TritStudio.Tests/Audit9Checks.cs','tests/TritStudio.Tests/Audit10Checks.cs','tests/TritStudio.Tests/Audit11Checks.cs','tests/TritStudio.Tests/Audit12Checks.cs','tests/TritStudio.Tests/Audit13Checks.cs','tests/TritStudio.Tests/Audit14Checks.cs','tests/TritStudio.Tests/Audit15Checks.cs','tests/TritStudio.Tests/Audit16Checks.cs','tests/TritStudio.Tests/Audit17Checks.cs'])
    result = {'status':'passed','scope':'Project/source/script structure and selected invariants, NOT compilation or execution',
              'projects':len(projects),'solution_projects':len(solution_projects),'csharp_source_files':len(cs_files),
              'core_test_cases_defined':core_tests,'python_modules_parsed':len(python_files),'bash_scripts_syntax_checked':len(sh_files),
              'dotnet_available':shutil.which('dotnet') is not None,'csharp_build':'NOT_RUN','headless_ui':'NOT_RUN','cuda':'NOT_RUN'}
    (ROOT/'reports/static-audit-v17.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':main()
