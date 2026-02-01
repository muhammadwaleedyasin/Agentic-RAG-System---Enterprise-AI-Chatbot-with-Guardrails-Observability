import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Input } from '@/components/ui/input';

describe('Input Component', () => {
  describe('Basic Functionality', () => {
    it('should render with default props', () => {
      render(<Input />);
      
      const input = screen.getByRole('textbox');
      expect(input).toBeInTheDocument();
      expect(input).toHaveClass('flex', 'h-10', 'w-full', 'rounded-md');
    });

    it('should render different input types correctly', () => {
      const { rerender } = render(<Input type="email" data-testid="email-input" />);
      let input = screen.getByTestId('email-input');
      expect(input).toHaveAttribute('type', 'email');

      rerender(<Input type="password" data-testid="password-input" />);
      input = screen.getByTestId('password-input');
      expect(input).toHaveAttribute('type', 'password');

      rerender(<Input type="number" data-testid="number-input" />);
      input = screen.getByTestId('number-input');
      expect(input).toHaveAttribute('type', 'number');

      rerender(<Input type="tel" data-testid="tel-input" />);
      input = screen.getByTestId('tel-input');
      expect(input).toHaveAttribute('type', 'tel');
    });

    it('should handle placeholder text', () => {
      render(<Input placeholder="Enter your name" />);
      
      const input = screen.getByPlaceholderText('Enter your name');
      expect(input).toBeInTheDocument();
      expect(input).toHaveClass('placeholder:text-secondary-500');
    });

    it('should handle default values', () => {
      render(<Input defaultValue="default text" />);
      
      const input = screen.getByDisplayValue('default text');
      expect(input).toBeInTheDocument();
    });

    it('should handle controlled values', () => {
      const { rerender } = render(<Input value="controlled value" onChange={() => {}} />);
      
      let input = screen.getByDisplayValue('controlled value');
      expect(input).toBeInTheDocument();

      rerender(<Input value="updated value" onChange={() => {}} />);
      input = screen.getByDisplayValue('updated value');
      expect(input).toBeInTheDocument();
    });
  });

  describe('Event Handling', () => {
    it('should handle onChange events', async () => {
      const handleChange = jest.fn();
      const user = userEvent.setup();
      
      render(<Input onChange={handleChange} />);
      
      const input = screen.getByRole('textbox');
      await user.type(input, 'test');
      
      expect(handleChange).toHaveBeenCalledTimes(4); // Once for each character
      expect(input).toHaveValue('test');
    });

    it('should handle onFocus and onBlur events', async () => {
      const handleFocus = jest.fn();
      const handleBlur = jest.fn();
      const user = userEvent.setup();
      
      render(<Input onFocus={handleFocus} onBlur={handleBlur} />);
      
      const input = screen.getByRole('textbox');
      
      await user.click(input);
      expect(handleFocus).toHaveBeenCalledTimes(1);
      expect(input).toHaveFocus();
      
      await user.tab();
      expect(handleBlur).toHaveBeenCalledTimes(1);
      expect(input).not.toHaveFocus();
    });

    it('should handle onKeyDown events', async () => {
      const handleKeyDown = jest.fn();
      const user = userEvent.setup();
      
      render(<Input onKeyDown={handleKeyDown} />);
      
      const input = screen.getByRole('textbox');
      await user.click(input);
      await user.keyboard('{Enter}');
      
      expect(handleKeyDown).toHaveBeenCalledTimes(1);
      expect(handleKeyDown).toHaveBeenCalledWith(
        expect.objectContaining({
          key: 'Enter',
        })
      );
    });
  });

  describe('Styling and Appearance', () => {
    it('should apply custom className', () => {
      render(<Input className="custom-class" />);
      
      const input = screen.getByRole('textbox');
      expect(input).toHaveClass('custom-class');
      expect(input).toHaveClass('flex'); // Should still have default classes
    });

    it('should have proper focus styles', () => {
      render(<Input />);
      
      const input = screen.getByRole('textbox');
      expect(input).toHaveClass('focus:outline-none');
      expect(input).toHaveClass('focus:ring-2');
      expect(input).toHaveClass('focus:ring-primary-500');
      expect(input).toHaveClass('focus:border-transparent');
    });

    it('should have proper disabled styling', () => {
      render(<Input disabled />);
      
      const input = screen.getByRole('textbox');
      expect(input).toBeDisabled();
      expect(input).toHaveClass('disabled:cursor-not-allowed');
      expect(input).toHaveClass('disabled:opacity-50');
    });

    it('should have proper border and background styling', () => {
      render(<Input />);
      
      const input = screen.getByRole('textbox');
      expect(input).toHaveClass('border');
      expect(input).toHaveClass('border-secondary-300');
      expect(input).toHaveClass('bg-white');
      expect(input).toHaveClass('dark:bg-secondary-900');
    });
  });

  describe('Accessibility', () => {
    it('should support ARIA attributes', () => {
      render(
        <Input
          aria-label="Search input"
          aria-describedby="search-help"
          aria-required="true"
        />
      );
      
      const input = screen.getByRole('textbox');
      expect(input).toHaveAttribute('aria-label', 'Search input');
      expect(input).toHaveAttribute('aria-describedby', 'search-help');
      expect(input).toHaveAttribute('aria-required', 'true');
    });

    it('should support required attribute', () => {
      render(<Input required />);
      
      const input = screen.getByRole('textbox');
      expect(input).toBeRequired();
    });

    it('should support readonly attribute', () => {
      render(<Input readOnly value="readonly text" />);
      
      const input = screen.getByDisplayValue('readonly text');
      expect(input).toHaveAttribute('readonly');
    });

    it('should be accessible via keyboard navigation', async () => {
      const user = userEvent.setup();
      
      render(
        <div>
          <Input data-testid="input1" />
          <Input data-testid="input2" />
        </div>
      );
      
      const input1 = screen.getByTestId('input1');
      const input2 = screen.getByTestId('input2');
      
      input1.focus();
      expect(input1).toHaveFocus();
      
      await user.tab();
      expect(input2).toHaveFocus();
    });
  });

  describe('Form Integration', () => {
    it('should work with form submission', () => {
      const handleSubmit = jest.fn((e) => e.preventDefault());
      
      render(
        <form onSubmit={handleSubmit}>
          <Input name="username" defaultValue="testuser" />
          <button type="submit">Submit</button>
        </form>
      );
      
      const input = screen.getByRole('textbox');
      const button = screen.getByRole('button');
      
      expect(input).toHaveAttribute('name', 'username');
      expect(input).toHaveValue('testuser');
      
      fireEvent.click(button);
      expect(handleSubmit).toHaveBeenCalledTimes(1);
    });

    it('should handle form validation', () => {
      render(<Input required minLength={3} maxLength={10} />);
      
      const input = screen.getByRole('textbox');
      expect(input).toHaveAttribute('required');
      expect(input).toHaveAttribute('minlength', '3');
      expect(input).toHaveAttribute('maxlength', '10');
    });
  });

  describe('Ref Forwarding', () => {
    it('should forward ref correctly', () => {
      const ref = React.createRef<HTMLInputElement>();
      render(<Input ref={ref} />);
      
      expect(ref.current).toBeInstanceOf(HTMLInputElement);
      expect(ref.current).toHaveClass('flex');
    });

    it('should allow ref methods to be called', () => {
      const ref = React.createRef<HTMLInputElement>();
      render(<Input ref={ref} />);
      
      expect(ref.current?.focus).toBeDefined();
      expect(ref.current?.blur).toBeDefined();
      expect(ref.current?.select).toBeDefined();
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty string values', () => {
      render(<Input value="" onChange={() => {}} />);
      
      const input = screen.getByRole('textbox');
      expect(input).toHaveValue('');
    });

    it('should handle special characters in values', async () => {
      const user = userEvent.setup();
      
      render(<Input />);
      
      const input = screen.getByRole('textbox');
      await user.type(input, '!@#$%^&*()');
      
      expect(input).toHaveValue('!@#$%^&*()');
    });

    it('should prevent input when disabled', async () => {
      const user = userEvent.setup();
      
      render(<Input disabled />);
      
      const input = screen.getByRole('textbox');
      await user.type(input, 'should not type');
      
      expect(input).toHaveValue('');
    });

    it('should handle multiple event handlers', async () => {
      const handleChange1 = jest.fn();
      const handleChange2 = jest.fn();
      const user = userEvent.setup();
      
      render(
        <Input 
          onChange={(e) => {
            handleChange1(e);
            handleChange2(e);
          }}
        />
      );
      
      const input = screen.getByRole('textbox');
      await user.type(input, 'a');
      
      expect(handleChange1).toHaveBeenCalledTimes(1);
      expect(handleChange2).toHaveBeenCalledTimes(1);
    });
  });
});