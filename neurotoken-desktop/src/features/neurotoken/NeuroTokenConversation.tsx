import { useEffect, useState, type ReactNode } from "react";
import { Composer } from "../composer/components/Composer";
import { Messages } from "../messages/components/Messages";
import type { ConversationItem } from "../../types";

type NeuroTokenConversationProps = {
  items: ConversationItem[];
  threadId: string | null;
  isLoading: boolean;
  isProcessing: boolean;
  draft: string;
  onDraftChange: (value: string) => void;
  onSend: (text: string) => void;
  readOnly?: boolean;
  children?: ReactNode;
};

export function NeuroTokenConversation({
  items,
  threadId,
  isLoading,
  isProcessing,
  draft,
  onDraftChange,
  onSend,
  readOnly = false,
  children,
}: NeuroTokenConversationProps) {
  const [processingStartedAt, setProcessingStartedAt] = useState<number | null>(null);

  useEffect(() => {
    if (isProcessing) {
      setProcessingStartedAt(Date.now());
    } else {
      setProcessingStartedAt(null);
    }
  }, [isProcessing, threadId]);

  return (
    <section className="neurotoken-conversation" aria-label="Task conversation">
      <div className="neurotoken-conversation-stream">
        <Messages
          items={items}
          threadId={threadId}
          workspaceId="neurotoken"
          isThinking={isProcessing}
          isLoadingMessages={isLoading}
          processingStartedAt={processingStartedAt}
          workspacePath={null}
          openTargets={[]}
          selectedOpenAppId=""
          showMessageFilePath={false}
        />
        {children}
      </div>
      <div className="neurotoken-conversation-composer">
        <Composer
          onSend={(text) => onSend(text)}
          onStop={() => undefined}
          canStop={false}
          disabled={!threadId || readOnly}
          appsEnabled={false}
          isProcessing={isProcessing}
          steerAvailable={false}
          followUpMessageBehavior="queue"
          composerFollowUpHintEnabled={false}
          collaborationModes={[]}
          selectedCollaborationModeId={null}
          onSelectCollaborationMode={() => undefined}
          models={[]}
          selectedModelId={null}
          onSelectModel={() => undefined}
          reasoningOptions={[]}
          selectedEffort={null}
          onSelectEffort={() => undefined}
          selectedServiceTier={null}
          reasoningSupported={false}
          accessMode="current"
          onSelectAccessMode={() => undefined}
          skills={[]}
          apps={[]}
          prompts={[]}
          files={[]}
          draftText={draft}
          onDraftChange={onDraftChange}
          historyKey={threadId ? `neurotoken:${threadId}` : null}
          sendLabel="Send"
        />
      </div>
    </section>
  );
}
